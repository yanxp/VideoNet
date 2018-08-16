# -*- coding: utf-8 -*-

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, 'data')
VIDEO_DIR = os.path.join(DATA_DIR, 'video')
IMAGE_DIR = os.path.join(DATA_DIR, 'image')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
import argparse
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--dataset', type=str, default='meitu', choices=['kinetics','meitu'])
parser.add_argument('--modality', type=str,default='RGB', choices=['RGB'])
parser.add_argument('--test_list', type=str,default='data/input.txt')
parser.add_argument('--weights', type=str,default='../mxnet/model/model.pth')
parser.add_argument('--num_class', type=int, default=50)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--save_name', type=str, default='data/output.txt')
parser.add_argument('--test_segments', type=int, default=3)
parser.add_argument('--new_length', type=int, default=1)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
