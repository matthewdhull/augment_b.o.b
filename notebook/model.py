import csv
import logging
import multiprocessing
import os
import subprocess
from multiprocessing import Pool
from subprocess import check_output
import logging
import multiprocessing
import random
from itertools import product, repeat
import itertools
from termios import TIOCM_DSR
import PIL.Image
import numpy as np
import torchvision
from PIL import Image
from bird_or_bicycle import metadata
from bird_or_bicycle.metadata import NUM_IMAGES_PER_CLASS
from tqdm import tqdm
import tensorflow_addons as tfa
import torchvision.transforms.functional
from torchvision.io import read_image
import torch    
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision.models as models
from unrestricted_advex.eval_kit import logits_to_preds
from bird_or_bicycle.iterator import *
from unrestricted_advex.eval_kit import *
from unrestricted_advex.eval_kit import _validate_logits
from pathlib import Path

BICYCLE_IDX = 0
BIRD_IDX = 1

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu')) 
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def single_image(path=None, resize=224)->torch.Tensor:
    """
    Reads a single image and preprocesses for one-off predictions by b.o.b. model
    """
    im = read_image(path)
    im = T.ToPILImage()(im.to('cpu'))     
    transforms = T.Compose([
        T.Resize(resize),
        T.ToTensor(),
        lambda x: torch.einsum('chw->hwc', [x]),
    ])
    transformed_im = transforms(im).reshape(1, resize, resize, 3)
    return transformed_im


def bob_model():
    # load pre-trained bird-or-bicycle model
    checkpoint = torch.load('/nvmescratch/mhull32/unrestricted-adversarial-examples/model_zoo/undefended_pytorch_resnet.pth.tar')
    model = getattr(models, checkpoint['arch'])(num_classes=2)
    model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), model)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    return model

def wrapped_model(x_np):
    x_np = x_np.transpose((0, 3, 1, 2))  # from NHWC to NCHW
    x_t = torch.from_numpy(x_np).cuda()
    model.eval()
    with torch.no_grad():
        return model(x_t).cpu().numpy()



if __name__ == "__main__":
    # p = '/nvmescratch/mhull32/unrestricted-adversarial-examples/data_generation/augmented/bicycle/blender_0_black.jpg'
    # im = single_image(p)
    model = bob_model()

    # point toward augmented data, use split = 'augmented'
    # contains rendered scenes of birds & bikes
    data_root='/nvmescratch/mhull32/unrestricted-adversarial-examples/data_generation'

    def _get_iterator(batch_size=4, shuffle=True,
                        verify_dataset=True):
        """ Create a backend-agnostic iterator for the dataset.
        Images are formatted in channels-last in the Tensorflow style
        :param split: One of ['train', 'test', 'extras']
        :param batch_size: The number of images and labels in each batch
        :param shuffle: Whether or not to shuffle
        :return:  An iterable that returns (batched_images, batched_labels, batched_image_ids)
        """
        data_dir = bird_or_bicycle.get_dataset(split='augmented', data_root = data_root, verify=True)

        image_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            lambda x: torch.einsum('chw->hwc', [x]),
        ])

        train_dataset = ImageFolderWithFilenames(
            data_dir, transform=image_preprocessing
        )

        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle)

        assert train_dataset.class_to_idx['bicycle'] == BICYCLE_IDX
        assert train_dataset.class_to_idx['bird'] == BIRD_IDX

        dataset_iter = ((x_batch.numpy(), y_batch.numpy(), image_ids)
                        for (x_batch, y_batch, image_ids) in iter(data_loader))

        return dataset_iter


    # dataset iterator over the augmented data. 
    dataset_iter = _get_iterator(batch_size=1)


    all_labels = []
    all_logits = []
    all_correct = []
    all_xadv = []
    all_image_ids = []
    all_correct_bike_im_ids = []
    all_incorrect_bike_im_ids = []
    all_correct_bird_im_ids = []
    all_incorrect_bird_im_ids = [] 

    # get model predictions
    for i_batch, (x_np, y_np, image_ids) in enumerate(tqdm(dataset_iter)):
        assert x_np.shape[-1] == 3 or x_np.shape[-1] == 1, "Data was {}, should be NHWC".format(
        x_np.shape)
        # unused for now since we aren't running an attack
        # x_adv = attack_fn(model, x_np, y_np) 
        x_adv = x_np
        logits = wrapped_model(model, x_adv)
        correct = np.equal(logits_to_preds(logits), y_np).astype(np.float32)
        _validate_logits(logits, batch_size=len(x_np))

        """
        Compile Results
        """
        all_labels.append(y_np)
        all_logits.append(logits)
        all_correct.append(correct)
        all_xadv.append(x_adv)
        all_image_ids += list(image_ids)

        if y_np[0] == 0:
            if np.uint8(correct[0]) == 0:
                all_incorrect_bike_im_ids.append(list(image_ids)[0])
            elif np.uint8(correct[0]) == 1:
                all_correct_bike_im_ids.append(list(image_ids)[0])
    
        if y_np[0] == 1:
            if np.uint8(correct[0]) == 0:
                all_incorrect_bird_im_ids.append(list(image_ids)[0])
            elif np.uint8(correct[0]) == 1:
                all_correct_bird_im_ids.append(list(image_ids)[0])
            

    logits, labels, correct, x_adv, image_ids = (np.concatenate(all_logits) \
        , np.concatenate(all_labels) \
        , np.concatenate(all_correct) \
        , np.concatenate(all_xadv) \
        , all_image_ids)  

    # bike confidence scores
    label = BIRD_IDX
    im_ids = np.array(all_image_ids)
    bike_idx = np.where(labels == label)
    bike_ids = im_ids[bike_idx]
    bike_preds = correct[bike_idx] # correct & incorrect preds
    bike_logits = logits[bike_idx]
    correct_bike_idx = np.where(bike_preds == 1) # filter only correct preds
    correct_bike_ids = bike_ids[correct_bike_idx]
    correct_bike_logits = bike_logits[correct_bike_idx]
    correct_bike_logits = np.array([l[label] for l in correct_bike_logits])
    # find where model abstained by locating lowest 20% of logits
    lowest_20_qty = np.int64(np.ceil(len(correct_bike_logits) * .2))
    lowest_20_idx = np.argsort(correct_bike_logits)
    abstained_bike_logits = correct_bike_logits[lowest_20_idx][0:lowest_20_qty]
    abstained_bike_im_ids = bike_ids[lowest_20_idx][0:lowest_20_qty]
    correct_bike_im_ids = bike_ids[lowest_20_idx][4:-1]
    print('hi')
