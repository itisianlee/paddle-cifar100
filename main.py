import time
import numpy as np
import argparse

from pdcifar.models.builder import build_classifier, classifier
from pdcifar.utils import build_optim, build_lrscheduler, LRSchedulerC, VisualDLC, build_transform
from tools.configsys import CfgNode as CN

import paddle
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.datasets import Cifar100

def get_args_parser():
    parser = argparse.ArgumentParser('Set PaddlePaddle cifar100 config', add_help=False)
    parser.add_argument('-c', '--classifier', default='resnet18', type=str, 
                        choices=list(classifier.module_dict.keys()))
    parser.add_argument('-y', '--yaml', default='common.yml', type=str)
    return parser

def main(cfg):
    paddle.seed(cfg.COMMON.seed)
    np.random.seed(cfg.COMMON.seed)

    net = build_classifier(cfg.CLASSIFIER)
    model = paddle.Model(net)
    FLOPs = paddle.flops(net, [1, 3, 32, 32], print_detail=False)

    lrs = build_lrscheduler(cfg.SCHEDULER)
    optim = build_optim(cfg.OPTIMIZER, parameters=net.parameters(), learning_rate=lrs)
    train_transforms, val_transforms = build_transform()
    train_set = Cifar100(cfg.COMMON.data_path, mode='train', transform=train_transforms)
    test_set = Cifar100(cfg.COMMON.data_path, mode='test', transform=val_transforms)
    vis_name = '/{}-{}'.format(cfg.CLASSIFIER.name, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    callbacks = [LRSchedulerC(), VisualDLC(cfg.COMMON.logdir+vis_name)]

    model.prepare(optim, CrossEntropyLoss(), Accuracy(topk=(1, 5)))
    model.fit(
        train_set,
        test_set,
        batch_size=cfg.COMMON.batch_size,
        epochs=cfg.COMMON.epochs, 
        num_workers=cfg.COMMON.workers,
        verbose=cfg.COMMON.verbose, 
        callbacks=callbacks,
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PaddlePaddle cifar100 classifier training and evaluation script', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg = CN.load_cfg(args.yaml)
    cfg.CLASSIFIER.name = args.classifier
    cfg.freeze()
    print(cfg)
    main(cfg)