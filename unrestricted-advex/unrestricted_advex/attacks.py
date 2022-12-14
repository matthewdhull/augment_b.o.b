from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging
import multiprocessing
import random
from itertools import product, repeat

import PIL.Image
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import torchvision.transforms.functional
from cleverhans.attacks import SPSA
from cleverhans.model import Model
from foolbox.attacks import BoundaryAttack as FoolboxBoundaryAttack
from imagenet_c import corrupt
from six.moves import xrange
from unrestricted_advex.utils import suppress_stdout

class Attack(object):
  name = None

  # TODO: Refactor this out of this object
  _stop_after_n_datapoints = None  # An attack can optionally run on only a subset of the dataset

  def __call__(self, *args, **kwargs):
    raise NotImplementedError()


class CleanData(Attack):
  """Also known as the "null attack". Just returns the unaltered clean image"""
  name = 'clean'

  def __call__(self, model_fn, images_batch_nhwc, y_np):
    del y_np, model_fn  # unused
    return images_batch_nhwc


def apply_transformation(x, angle, dx, dy, black_border_size):
  """Apply a transformation to an image"""
  x = PIL.Image.fromarray((x * 255.).astype(np.uint8))

  # Apply black border
  orig_width = x.width
  orig_height = x.height

  padding_x = int(float(orig_height) * black_border_size)
  padding_y = int(float(orig_width) * black_border_size)

  x = torchvision.transforms.functional.resize(x, [
    orig_height - (padding_y * 2),
    orig_width - (padding_x * 2),
  ])

  x = torchvision.transforms.functional.pad(x, (
    padding_x,
    padding_y,
  ))

  # Apply rotation and transformation
  x = torchvision.transforms.functional.affine(
    x,
    angle=angle,
    translate=[dx, dy],
    shear=0,
    resample=PIL.Image.BILINEAR,
    scale=1
  )
  return np.array(x).astype(np.float32) / 255.


def _apply_transformation_star(args):
  """python 2.7 doesn't support Pool.starmap"""
  return apply_transformation(*args)


def _batched_apply(fn, x, batch_size):
  all_out = []
  n_batches = int(np.ceil(len(x) / batch_size))
  for batch_idx in range(n_batches):
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch = x[batch_start: batch_end]
    all_out.append(fn(batch))
  return np.concatenate(all_out)


class SimpleSpatialAttack(Attack):
  """Fast attack from "A Rotation and a Translation Suffice: Fooling CNNs with
    Simple Transformations", Engstrom et al. 2018

    https://arxiv.org/pdf/1712.02779.pdf
    """
  name = 'spatial_grid'

  def __init__(self,
               spatial_limits,
               grid_granularity,
               black_border_frac,
               ):
    self.spatial_limits = spatial_limits
    self.grid_granularity = grid_granularity
    self.black_border_size = black_border_frac

    # This task is CPU bound, so we try to use all the CPUs
    N_WORKERS = multiprocessing.cpu_count() or 1
    logging.info("SpatialAttack will run  with %s cpus" % N_WORKERS)
    if N_WORKERS < 8:
      print("WARNING: Running SpatialAttack with fewer than 8 CPUs "
            "is extremely slow because SpatialAttack is CPU-bound")
    self.pool = multiprocessing.Pool(N_WORKERS)

  def __call__(self, model_fn, images_batch_nhwc, y_np):
    batch_size = len(images_batch_nhwc)

    dx_limit, dy_limit, angle_limit = self.spatial_limits
    n_dxs, n_dys, n_angles = self.grid_granularity

    # Define the range of transformations
    dxs = np.linspace(-dx_limit, dx_limit, n_dxs)
    dys = np.linspace(-dy_limit, dy_limit, n_dxs)
    angles = np.linspace(-angle_limit, angle_limit, n_angles)

    transforms = list(itertools.product(*[dxs, dys, angles]))

    # Keep track of the worst corruption for each image
    worst_xs = np.copy(images_batch_nhwc)
    worst_loss = [np.NINF] * batch_size

    # Iterate through each image in the batch
    for batch_idx, x in enumerate(images_batch_nhwc):
      transform_args = [(x, angle, dx, dy, self.black_border_size)
                        for (angle, dx, dy) in transforms]

      adv_x_batch = self.pool.map(_apply_transformation_star, transform_args)
      adv_x_batch = [x for x in adv_x_batch]

      logits_batch = _batched_apply(model_fn, np.array(adv_x_batch),
                                    batch_size=32)
      label = int(y_np[batch_idx])

      # This is left un-vectorized for readability
      for (logits, adv_x) in zip(logits_batch, adv_x_batch):
        correct_logit, wrong_logit = logits[label], logits[1 - label]

        # We can choose different loss functions to optimize in the
        # attack. For now, optimize the magnitude of the wrong logit
        # because we use this as our confidence threshold
        loss = wrong_logit - correct_logit
        if loss > worst_loss[batch_idx]:
          worst_xs[batch_idx] = adv_x
          worst_loss[batch_idx] = loss
    return worst_xs


class SpsaAttack(Attack):
  name = 'spsa'

  def __init__(self, model, image_shape_hwc, epsilon=(16. / 255),
               num_steps=200, batch_size=32, is_debug=False):
    self.graph = tf.Graph()

    with self.graph.as_default():
      self.sess = tf.Session(graph=self.graph)

      self.x_input = tf.placeholder(tf.float32, shape=(1,) + image_shape_hwc)
      self.y_label = tf.placeholder(tf.int32, shape=(1,))

      self.model = model
      attack = SPSA(CleverhansPyfuncModelWrapper(self.model), sess=self.sess)
      self.x_adv = attack.generate(
        self.x_input,
        y=self.y_label,
        epsilon=epsilon,
        num_steps=num_steps,
        early_stop_loss_threshold=-1.,
        batch_size=batch_size,
        is_debug=is_debug)

    self.graph.finalize()

  def __call__(self, model, x_np, y_np):  # (4. / 255)):
    if model != self.model:
      raise ValueError('Cannot call spsa attack on different models')
    del model  # unused except to check that we already wired it up right

    with self.graph.as_default():
      all_x_adv_np = []
      for i in xrange(len(x_np)):
        x_adv_np = self.sess.run(self.x_adv, feed_dict={
          self.x_input: np.expand_dims(x_np[i], axis=0),
          self.y_label: np.expand_dims(y_np[i], axis=0),
        })
        all_x_adv_np.append(x_adv_np)
      return np.concatenate(all_x_adv_np)


def corrupt_float32_image(x, corruption_name, severity):
  """Convert to uint8 and back to conform to corruption API"""
  x = np.copy(x)  # We make a copy to avoid changing things in-place
  x = (x * 255).astype(np.uint8)

  corrupt_x = corrupt(
    x,
    corruption_name=corruption_name,
    severity=severity)
  return corrupt_x.astype(np.float32) / 255.


def _corrupt_float32_image_star(args):
  return corrupt_float32_image(*args)


class CommonCorruptionsAttack(Attack):
  name = "common_corruptions"

  def __init__(self, severity=1):
    self.corruption_names = [
      'gaussian_noise',
      'shot_noise',
      'impulse_noise',
      'defocus_blur',
      'glass_blur',
      'motion_blur',
      'zoom_blur',
      # 'snow', # Snow does not work in python 2.7
      # 'frost', # Frost is not working correctly
      'fog',
      'brightness',
      'contrast',
      'elastic_transform',
      'pixelate',
      'jpeg_compression',
      'speckle_noise',
      'gaussian_blur',
      'spatter',
      'saturate']
    self.severity = severity
    self.pool = multiprocessing.Pool(len(self.corruption_names))

  def __call__(self, model_fn, images_batch_nhwc, y_np):
    assert images_batch_nhwc.shape[1:] == (224, 224, 3), \
      "Image shape must equal (N, 224, 224, 3)"
    batch_size = len(images_batch_nhwc)

    # Keep track of the worst corruption for each image
    worst_corruption = np.copy(images_batch_nhwc)
    worst_loss = [np.NINF] * batch_size

    # Iterate through each image in the batch
    for batch_idx, x in enumerate(images_batch_nhwc):
      corrupt_args = [(x, corruption_name, self.severity)
                      for corruption_name in self.corruption_names]
      corrupt_x_batch = self.pool.map(_corrupt_float32_image_star, corrupt_args)
      logits_batch = model_fn(np.array(corrupt_x_batch))
      label = y_np[batch_idx]

      # This is left un-vectorized for readability
      for (logits, corrupt_x) in zip(logits_batch, corrupt_x_batch):
        correct_logit, wrong_logit = logits[label], logits[1 - label]

        # We can choose different loss functions to optimize in the
        # attack. For now, optimize the magnitude of the wrong logit
        # because we use this as our confidence threshold
        loss = wrong_logit
        # loss = wrong_logit - correct_logit

        if loss > worst_loss[batch_idx]:
          worst_corruption[batch_idx] = corrupt_x
          worst_loss[batch_idx] = loss

    return worst_corruption


class BoundaryAttack(Attack):
  name = "boundary"

  def __init__(self, model, image_shape_hwc, max_l2_distortion=4, label_to_examples=None):
    if label_to_examples is None:
      label_to_examples = {}

    self.max_l2_distortion = max_l2_distortion

    class Model:
      def bounds(self):
        return [0, 1]

      def predictions(self, img):
        return model(img[np.newaxis, :, :, :])[0]

      def batch_predictions(self, img):
        return model(img)

    self.label_to_examples = label_to_examples

    h, w, c = image_shape_hwc
    mse_threshold = max_l2_distortion ** 2 / (h * w * c)
    try:
      # Foolbox 1.5 allows us to use a threshold the attack will abort after
      # reaching. Because we only care about a distortion of less than 4, as soon
      # as we reach it, we can just abort and move on to the next image.
      self.attack = FoolboxBoundaryAttack(model=Model(), threshold=mse_threshold)
    except:
      # Fall back to the original implementation.
      print("WARNING: Using foolbox version < 1.5 will cause the "
            "boundary attack to perform more work than is required. "
            "Please upgrade to version 1.5")
      self.attack = FoolboxBoundaryAttack(model=Model())

  def __call__(self, model, x_np, y_np):
    r = []
    for i in range(len(x_np)):
      other = 1 - y_np[i]
      initial_adv = random.choice(self.label_to_examples[other])
      try:
        with suppress_stdout():  # Foolbox is extremely verbose, so we silence it
          adv = self.attack(x_np[i], y_np[i],
                            log_every_n_steps=100,  # Reduce verbosity of the attack
                            starting_point=initial_adv
                            )
        distortion = np.sum((x_np[i] - adv) ** 2) ** .5
        if distortion > self.max_l2_distortion:
          # project to the surface of the L2 ball
          adv = x_np[i] + (adv - x_np[i]) / distortion * self.max_l2_distortion

      except AssertionError as error:
        if str(error).startswith("Invalid starting point provided."):
          print("WARNING: The model misclassified the starting point (the target) "
                "from BoundaryAttack. This means that the attack will fail on this "
                "specific point (but is likely to succeed on other points.")
          adv = x_np[i]  # Just return the non-adversarial point
        else:
          raise error

      r.append(adv)
    return np.array(r)


class SpatialGridAttack(Attack):
  """Attack from "A Rotation and a Translation Suffice: Fooling CNNs with
  Simple Transformations", Engstrom et al. 2018

  https://arxiv.org/pdf/1712.02779.pdf
  """
  name = 'spatial_grid'

  def __init__(self, image_shape_hwc,
               spatial_limits,
               grid_granularity,
               black_border_size,
               valid_check=None,
               ):
    """
    :param model_fn: a callable: batch-input -> batch-probability in [0, 1]
    :param spatial_limits:
    :param grid_granularity:
    """
    self.limits = spatial_limits
    self.granularity = grid_granularity
    self.valid_check = valid_check

    # Construct graph for spatial attack
    self.graph = tf.Graph()
    with self.graph.as_default():
      self._x_for_trans = tf.placeholder(tf.float32, shape=[None] + list(image_shape_hwc))
      self._t_for_trans = tf.placeholder(tf.float32, shape=[None, 3])

      x = apply_black_border(
        self._x_for_trans,
        image_height=image_shape_hwc[0],
        image_width=image_shape_hwc[1],
        border_size=black_border_size
      )

      self._tranformed_x_op = tf_apply_transformation(
        x,
        transform=self._t_for_trans,
        image_height=image_shape_hwc[0],
        image_width=image_shape_hwc[1],
      )
      self.session = tf.Session()

    self.grid_store = []

  def __call__(self, model_fn, x_np, y_np):
    n = len(x_np)
    grid = product(*list(np.linspace(-l, l, num=g)
                         for l, g in zip(self.limits, self.granularity)))

    worst_x = np.copy(x_np)
    max_xent = np.zeros(n)
    all_correct = np.ones(n).astype(bool)

    trans_np = np.stack(
      repeat([0, 0, 0], n))
    with self.graph.as_default():
      x_downsize_np = self.session.run(self._tranformed_x_op, feed_dict={
        self._x_for_trans: x_np,
        self._t_for_trans: trans_np,

      })

    for horizontal_trans, vertical_trans, rotation in grid:
      trans_np = np.stack(
        repeat([horizontal_trans, vertical_trans, rotation], n))

      # Apply the spatial attack
      with self.graph.as_default():
        x_np_trans = self.session.run(self._tranformed_x_op, feed_dict={
          self._x_for_trans: x_np,
          self._t_for_trans: trans_np,
        })
      # See how the model_fn performs on the perturbed input
      logits = model_fn(x_np_trans)
      preds = np.argmax(logits, axis=1)

      cur_xent = _sparse_softmax_cross_entropy_with_logits_from_numpy(
        logits, y_np, self.graph, self.session)

      cur_xent = np.asarray(cur_xent)
      cur_correct = np.equal(y_np, preds)

      if self.valid_check:
        is_valid = self.valid_check(x_downsize_np, x_np_trans)
        cur_correct |= ~is_valid
        cur_xent -= is_valid * 1e9

      # Select indices to update: we choose the misclassified transformation
      # of maximum xent (or just highest xent if everything else if correct).
      idx = (cur_xent > max_xent) & (cur_correct == all_correct)
      idx = idx | (cur_correct < all_correct)
      max_xent = np.maximum(cur_xent, max_xent)
      all_correct = cur_correct & all_correct

      idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1)

      idx = np.expand_dims(idx, axis=-1)
      idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
      worst_x = np.where(idx, x_np_trans, worst_x, )  # shape (bsize, 32, 32, 3)

    return worst_x


def _sparse_softmax_cross_entropy_with_logits_from_numpy(logits_np, labels_np, graph, sess):
  """Helper that calls the TF sparse_softmax_cross_entropy_with_logits function"""
  with graph.as_default():
    labels_tf = tf.placeholder(tf.int32, [None])
    logits_tf = tf.placeholder(tf.float32, [None, None])
    xent_tf = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_tf, logits=logits_tf)
    return sess.run(xent_tf, feed_dict={
      labels_tf: labels_np, logits_tf: logits_np})


def apply_black_border(x, image_height, image_width, border_size):
  x = tf.image.resize(x, (image_width - border_size,
                                 image_height - border_size))
  x = tf.pad(x, [[0, 0],
                 [border_size, border_size],
                 [border_size, border_size],
                 [0, 0]], 'CONSTANT')
  return x


def tf_apply_transformation(x, transform, image_height, image_width):
  # Map a transformation onto the input
  trans_x, trans_y, rot = tf.unstack(transform, axis=1)
  rot *= np.pi / 180  # convert degrees to radians

  # Pad the image to prevent two-step rotation / translation
  # resulting in a cropped image
  x = tf.pad(x, [[0, 0], [50, 50], [50, 50], [0, 0]], 'CONSTANT')

  # rotate and translate image
  ones = tf.ones(shape=tf.shape(trans_x))
  zeros = tf.zeros(shape=tf.shape(trans_x))
  trans = tf.stack([ones, zeros, -trans_x,
                    zeros, ones, -trans_y,
                    zeros, zeros], axis=1)
  x = tfa.image.rotate(x, rot, interpolation='BILINEAR')
  x = tfa.image.transform(x, trans, interpolation='BILINEAR')
  return tf.image.resize_with_crop_or_pad(
    x, image_height, image_width)


class CleverhansPyfuncModelWrapper(Model):
  nb_classes = 2
  num_classes = 2

  def __init__(self, model_fn):
    """
    Wrap a callable function that takes a numpy array of shape (N, C, H, W),
    and outputs a numpy vector of length N, with each element in range [0, 1].
    """
    self.model_fn = model_fn

  def fprop(self, x, **kwargs):
    logits_op = tf.py_func(self.model_fn, [x], tf.float32)
    return {'logits': logits_op}


class RandomSpatialAttack(Attack):
  """Apply a single random rotation and translation
  as in "A Rotation and a Translation Suffice: Fooling CNNs with
  Simple Transformations", Engstrom et al. 2018

  https://arxiv.org/pdf/1712.02779.pdf
  """
  name = 'random_spatial'

  def __init__(self, image_shape_hwc, spatial_limits, black_border_size, valid_check=None):
    self.limits = spatial_limits
    self.valid_check = valid_check

    # Construct graph for spatial attack
    self.graph = tf.Graph()
    with self.graph.as_default():
      self._x_for_trans = tf.compat.v1.placeholder(tf.float32, shape=[None] + list(image_shape_hwc))
      self._t_for_trans = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])

      x = apply_black_border(
        self._x_for_trans,
        image_height=image_shape_hwc[0],
        image_width=image_shape_hwc[1],
        border_size=black_border_size
      )

      self._tranformed_x_op = tf_apply_transformation(
        x,
        transform=self._t_for_trans,
        image_height=image_shape_hwc[0],
        image_width=image_shape_hwc[1],
      )
      self.session = tf.Session()

  def __call__(self, model_fn, x_np, y_np):
    # randomize each example separately

    with self.graph.as_default():
      result = np.zeros(x_np.shape, dtype=x_np.dtype)
      did = np.zeros(x_np.shape[0], dtype=np.bool)

      trans_np = np.stack(
        repeat([0, 0, 0], x_np.shape[0]))
      x_downsize_np = self.session.run(self._tranformed_x_op, feed_dict={
        self._x_for_trans: x_np,
        self._t_for_trans: trans_np,
      })

      while True:
        random_transforms = (np.random.uniform(-lim, lim, len(x_np)) for lim in self.limits)
        trans_np = np.stack(random_transforms, axis=1)
        out = self.session.run(self._tranformed_x_op, feed_dict={
          self._x_for_trans: x_np,
          self._t_for_trans: trans_np,
        })

        if self.valid_check is None:
          return out
        else:
          ok = self.valid_check(x_downsize_np, out)
          result[ok] = out[ok]
          did[ok] = True
          if np.all(did):
            return result


class SpsaWithRandomSpatialAttack(Attack):
  """Apply a single random rotation and translation and then apply SPSA
  to the resulting image
  """
  name = "spsa_with_random_spatial"

  def __init__(self, model, image_shape_hwc, spatial_limits, black_border_size,
               epsilon=(16. / 255), num_steps=32, is_debug=False,
               valid_check=None):
    self.random_spatial_attack = RandomSpatialAttack(
      image_shape_hwc,
      valid_check=valid_check,
      spatial_limits=spatial_limits,
      black_border_size=black_border_size)

    self.spsa_attack = SpsaAttack(
      model,
      image_shape_hwc,
      epsilon=epsilon,
      num_steps=num_steps,
      batch_size=64,  # this is number of samples in the new cleverhans
      is_debug=is_debug)

  def __call__(self, model, x_np, y_np):
    x_after_spatial_np = self.random_spatial_attack(model, x_np, y_np)
    x_adv = self.spsa_attack(model, x_after_spatial_np, y_np)
    return x_adv


class BoundaryWithRandomSpatialAttack(Attack):
  """Apply a single random rotation and translation and then apply SPSA
  to the resulting image
  """
  name = "boundary_with_random_spatial"

  def __init__(self, model, image_shape_hwc, spatial_limits, black_border_size,
               max_l2_distortion=4, label_to_examples=None, valid_check=None):
    self.random_spatial_attack = RandomSpatialAttack(
      image_shape_hwc,
      valid_check=valid_check,
      spatial_limits=spatial_limits,
      black_border_size=black_border_size)

    self.boundary_attack = BoundaryAttack(
      model,
      max_l2_distortion=max_l2_distortion,
      image_shape_hwc=image_shape_hwc,
      label_to_examples=label_to_examples)

  def __call__(self, model, x_np, y_np):
    x_after_spatial_np = self.random_spatial_attack(model, x_np, y_np)
    x_adv = self.boundary_attack(model, x_after_spatial_np, y_np)
    return x_adv
