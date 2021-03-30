import time
import numpy as np
import argparse

from pdcifar.models.builder import build_classifier, classifier
from pdcifar.utils import build_optim, build_lrscheduler, LRSchedulerC, VisualDLC, build_transform

import paddle
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.datasets import Cifar100

def get_args_parser():
    parser = argparse.ArgumentParser('Set PaddlePaddle cifar100 config', add_help=False)
    parser.add_argument('-c', '--classifier', default='resnet18', type=str, 
                        choices=list(classifier.module_dict.keys()))
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--data_path', default='data/cifar-100-python.tar.gz', type=str)
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--verbose', default=1, type=int, choices=[0, 1, 2],
                        help='The verbosity mode, should be 0, 1, or 2. 0 = silent, 1 = progress bar, \
                              2 = one line per epoch. Default: 1.')
    parser.add_argument('--seed', default=2021, type=int,
                        help='Seed for initializing training')
    parser.add_argument('--optim', default='momentum', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "optim"')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum of SGD solver')
    parser.add_argument('--lrscheduler', default='cosine', type=str,
                        help='Learning rate scheduler')
    parser.add_argument('--warm_up_step', default=2000, type=int, 
                        help='Warm up step')
    return parser

def main(args):
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    net = build_classifier(args.classifier)
    model = paddle.Model(net)

    lrs = build_lrscheduler(args.lrscheduler, args.lr, args.warm_up_step, T_max=args.epochs)
    optim = build_optim(name='momentum', parameters=net.parameters(), learning_rate=lrs, 
                        momentum=args.momentum, weight_decay=args.weight_decay)
    train_transforms, val_transforms = build_transform()
    train_set = Cifar100(args.data_path, mode='train', transform=train_transforms)
    test_set = Cifar100(args.data_path, mode='test', transform=val_transforms)
    vis_name = '/{}-{}'.format(args.classifier, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    callbacks = [LRSchedulerC(), VisualDLC(args.logdir+vis_name)]

    model.prepare(optim, CrossEntropyLoss(), Accuracy(topk=(1, 5)))
    model.fit(
        train_set,
        test_set,
        batch_size=args.batch_size,
        epochs=args.epochs, 
        num_workers=args.workers,
        verbose=args.verbose, 
        callbacks=callbacks,
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PaddlePaddle cifar100 classifier training and evaluation script', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)