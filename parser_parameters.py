from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def train_argument_parser():
    parser = argparse.ArgumentParser(description='MAMP')

    parser.add_argument('--proc_name', type=str, default='VOS',
                        help='proc name')
    parser.add_argument('--arch', type=str, default='MAMP',
                        help='arch name')
    parser.add_argument('--train_corr_radius', type=int, default=6,
                        help='local corr radius in training')
    parser.add_argument('--ref_num', type=int, default=2,
                        help='num of frames in one forward')
    parser.add_argument('--is_amp', action='store_true',
                        help='use mixed precision')
    parser.add_argument('--img_size', type=int, default=256,
                        help='image size')

    # Data options
    parser.add_argument('--datapath', default='../dataset/YOUTUBE/all',
                        help='Data path for Kinetics')
    parser.add_argument('--savepath', type=str, default='ckpt',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')

    # Training options
    parser.add_argument('--epochs', type=int, default=35,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--bsize', type=int, default=24,
                        help='batch size for training')
    parser.add_argument('--worker', type=int, default=24,
                        help='number of dataloader threads')

    return parser


def test_argument_parser():
    parser = argparse.ArgumentParser(description='MAMP')

    parser.add_argument('--arch', type=str, default='MAMP',
                        help='arch name')
    parser.add_argument('--memory_length', type=int, default=5)
    parser.add_argument('--pad_divisible', type=int, default=1,
                        help='pad images to be divisible by n')
    parser.add_argument('--test_corr_radius', type=int, default=12,
                        help='local corr radius in testing')
    parser.add_argument('--optical_flow_warp', type=int, default=0)

    parser.add_argument('--datapath', help='Data path for Davis')
    parser.add_argument('--savepath', type=str, default='ckpt',
                        help='Path for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint file to resume')
    parser.add_argument('--proc_name', type=str, default='VOS',
                        help='proc name')
    parser.add_argument('--is_amp', action='store_true', help='use mixed precision')

    return parser
