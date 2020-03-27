from pathlib import Path

import numpy as np
import torch

from TSMLDG import MetaFrameWork
from dataset.dg_dataset import get_target_loader
from network.components.customized_evaluate import AverageMeter

import argparse

parser = argparse.ArgumentParser(description='TSMLDG test args parser')
parser.add_argument('--name', default='exp', help='name of the experiment')
parser.add_argument('--target', default='C', help='target domain strins : GSMIC, it can be multi targets')
parser.add_argument('--normal-eval', type=bool, default=False, help='normally eval the network')
parser.add_argument('--target-eval', type=bool, default=True, help='target-specific eval the network')
parser.add_argument('--test-size', type=int, default=16, help='the batch size of target specific normalization')


def test_one_run(framework, run_name, targets, batches=[16], normal_eval=True, target_specific_eval=True):
    # for one experiment, test multi targets in multi batch_sizes
    print('=' * 20 + ' {} '.format(run_name) + '=' * 20)
    framework.save_path = Path(run_name)
    framework.load('best_city', strict=False)

    seeds = [12, 123, 1234, 12345, 123456]
    for target in targets:
        print('-' * 20 + ' {} '.format(target) + '-' * 20)
        iou = AverageMeter()

        if normal_eval:
            framework.val(get_target_loader(target, 8, shuffle=False))

        if target_specific_eval:
            for batch in batches:
                print('------ {} in 5 runs ------'.format(batch))
                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    res = framework.target_specific_val(get_target_loader(target, batch, shuffle=True))
                    iou.update(res)
                print('mean, batch : {},  mIoU : {},'.format(batch, iou.avg))
                iou.reset()


def do_lots_of_exp_tests(names=['exp'], targets=['C'], batch_sizes=[[16]], **kwargs):
    framework = MetaFrameWork(name='tmp')
    for name, target, batch in zip(names, targets, batch_sizes):
        test_one_run(framework, name, target, batches=batch, **kwargs)


def parse():
    args = parser.parse_args()
    names = [args.name]
    targets = [args.target]
    for name in args.target:
        assert name in 'GSIMCcuv'
    batches = [[args.test_size]]
    do_lots_of_exp_tests(names, targets, batches, normal_eval=args.normal_eval, target_specific_eval=args.target_eval)


if __name__ == '__main__':
    parse()

