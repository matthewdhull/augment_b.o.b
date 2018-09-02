"""
python pytorch_undefended.py --evaluate

train on extras, test on train:
without stage-wise lr decay: Prec@1 90.900 at epoch 89
with stage-wise lr decay: Prec@1 93.400 at epoch 89

train on extras+train, test on test: Prec@1 95.600

train with resnet 50, weight decay 5e-4: Prec@1 97.300
"""
import argparse
import os
import shutil
import time
import warnings

import matplotlib
import numpy as np
import bird_or_bicyle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

matplotlib.use('PDF')
from matplotlib import pyplot as plt

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--predict', dest='predict', action='store_true',
                    help='Compute and output predictions.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_prec1 = 0


def main():
  global args, best_prec1
  args = parser.parse_args()

  args.lr = 0.1 * (args.batch_size / 256)
  args.workers = int(4 * (args.batch_size / 256))

  if args.data == '':
    args.data = bird_or_bicyle.dataset.default_data_root()
    # args.data = "/root/datasets/unpacked_imagenet_pytorch"

  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
                  'disable data parallelism.')

  # create model
  model = models.resnet50(num_classes=2, pretrained=args.pretrained)

  if args.gpu is not None:
    model = model.cuda(args.gpu)
  else:
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
      model.features = torch.nn.DataParallel(model.features)
      model.cuda()
    else:
      model = torch.nn.DataParallel(model).cuda()

  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)

  optimizer = torch.optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 80], gamma=0.2)

  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint['epoch']
      best_prec1 = checkpoint['best_prec1']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  # Data loading code
  traindirs = [os.path.join(args.data, partition)
               for partition in ['train', 'extras']]
  valdir = os.path.join(args.data, 'test')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  train_dataset = [datasets.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ]))
    for traindir in traindirs]
  if len(train_dataset) == 1:
    train_dataset = train_dataset[0]
  else:
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)

  # Duplicated the dataset to make it as
  # train_dataset.samples = train_dataset.samples * 100

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

  val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

  if args.predict:
    predict(val_loader, model)
    return

  if args.evaluate:
    validate(val_loader, model, criterion)

  for epoch in range(args.start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)
    lr_scheduler.step()

    # evaluate on validation set
    if args.evaluate:
      prec1 = validate(val_loader, model, criterion)

      # remember best prec@1 and save checkpoint
      is_best = prec1 > best_prec1
      best_prec1 = max(prec1, best_prec1)
      save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
      }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, (x, target) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if args.gpu is not None:
      x = x.cuda(args.gpu, non_blocking=True)
    target = target.cuda(args.gpu, non_blocking=True)

    # compute output
    output = model(x)
    loss = criterion(output, target)

    # measure accuracy and record loss
    prec1 = accuracy(output, target)

    losses.update(loss.item(), x.size(0))
    top1.update(prec1.item(), x.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t)'.format(
        epoch, i, len(train_loader), batch_time=batch_time,
        data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
      if args.gpu is not None:
        input = input.cuda(args.gpu, non_blocking=True)
      target = target.cuda(args.gpu, non_blocking=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec1 = accuracy(output, target)
      losses.update(loss.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % args.print_freq == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
          i, len(val_loader), batch_time=batch_time, loss=losses,
          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

  return top1.avg


def predict(val_loader, model):
  model.eval()
  with torch.no_grad():
    outputs_all = []
    targets_all = []
    for inputs, targets in val_loader:
      if args.gpu is not None:
        inputs = inputs.cuda(args.gpu, non_blocking=True)
      targets = targets.cuda(args.gpu, non_blocking=True)

      outputs = model(inputs)
      outputs_all.append(outputs.cpu().numpy())
      targets_all.append(targets.cpu().numpy())

    outputs_all = np.concatenate(outputs_all)
    targets_all = np.concatenate(targets_all)
    np.savez('predictions.npz', predictions=outputs_all, targets=targets_all)

    # TODO: agree on model API and how confidences are computed
    predictions_01 = np.argmax(outputs_all, axis=1)
    confidences = np.abs(outputs_all[:, 0] - outputs_all[:, 1])
    idxs = get_cov_to_confident_error_idxs(predictions_01, confidences,
                                           targets_all)
    plt.figure()
    plot_cov_to_confident_error_idxs(idxs, len(predictions_01))
    plt.savefig('cov_to_confident_error_curve.pdf')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  lr = args.lr * (0.1 ** (epoch // 30))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def accuracy(output, target):
  """Computes the precision@k for the specified values of k"""
  with torch.no_grad():
    batch_size = target.size(0)
    pred = torch.argmax(output, dim=1)
    correct = pred.eq(target)
    num_correct = correct.float().sum(0)
    return num_correct.mul_(100.0 / batch_size)


if __name__ == '__main__':
  main()
