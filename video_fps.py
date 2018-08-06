import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
import os
from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
import time
from PIL import Image
from torch.autograd import Variable
import collections
import random
# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics','meitu'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
elif args.dataset == 'meitu':
    num_class = 50
else:
    raise ValueError('Unknown dataset '+args.dataset)

class VideoRecord(object):
    def __init__(self,row):
        self._data = row
    @property
    def path(self):
        return self._data[0]
    @path.setter
    def path(self,vidpath):
        self._data[0] = vidpath
    @property
    def num_frames(self):
        return int(self._data[1])
    @num_frames.setter
    def num_frames(self,number):
        self._data[1] = str(number)
    @property
    def label(self):
        return int(self._data[2])
    @label.setter
    def label(self,num):
        self._data[2] = str(num)

video_list = [VideoRecord(x.strip().split(',')) for x in open(args.test_list)] 
frames_dir = 'tmp/'
if not os.path.isdir(frames_dir):
    os.makedirs(frames_dir)

class TSNDataset(object):
    def __init__(self,args,new_length,transform,net,num_class):
        self.args = args
        self.new_length = new_length
        self.transform = transform
        self.net = net
        self.num_class = num_class
    def get_test_indices(self,vidRecord):
        tick = (vidRecord.num_frames - self.new_length + 1 ) /float(self.args.test_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.args.test_segments)])
        return offsets+1

    def load_image(self,path,idx):
        return [Image.open(os.path.join(path, 'img_{:05d}.jpg'.format(idx))).convert('RGB')] 
   
    def get(self,vidRecord,offsets):
        images = list()
        for ind in offsets:
            p = int(ind)
            for i in range(self.new_length):
                img = self.load_image(vidRecord.path,p)
                images.extend(img)
                if p<vidRecord.num_frames:
                    p += 1
        return images        
   
    def run_ffmpeg(self,vidRecord,frames_dir): 
       # path = os.path.join(frames_dir,vidRecord.path.split('/')[-1].split('.')[0])
       # if not os.path.isdir(path):
       #     os.makedirs(path)
        path = frames_dir
        str_video= "ffmpeg -i "+" " + vidRecord.path + " "+"-r 10 -q:v 2 "+path+"/img_%05d.jpg"
        os.system(str_video)
        return path
    def detect_video(self,vidRecord,frames_dir):
        # todo accelate
        path = self.run_ffmpeg(vidRecord,frames_dir)
        cnt_img = len(os.listdir(path))
        vidRecord.num_frames = cnt_img
        vidRecord.path = path
        offsets = self.get_test_indices(vidRecord)
        images = self.get(vidRecord,offsets)
        data = self.transform(images)
        data = data.unsqueeze(0)
        input_var = Variable(data.view(-1, 3, data.size(2), data.size(3)),volatile=True)
        rst = self.net(input_var).data.cpu().numpy().copy()
        vid_pred = rst.reshape((self.args.test_crops, self.args.test_segments, self.num_class)).mean(axis=0).reshape((self.args.test_segments, 1, self.num_class))
        vid_pred_max = np.argmax(np.mean(vid_pred, axis=0))
        return vid_pred_max

net = TSN(num_class, 1, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}

net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))
transform=torchvision.transforms.Compose([
    cropping,
    Stack(roll=args.arch == 'BNInception'),
    ToTorchFormatTensor(div=args.arch != 'BNInception'),
    GroupNormalize(net.input_mean, net.input_std),])

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))
net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()

Meitu = TSNDataset(args,1,transform,net,num_class)
pred_dict = collections.defaultdict(int)
proc_start_time = time.time()
total = 0
correct = 0
for i,vid in enumerate(video_list):
    prefix = vid.path.split('/')[-1].split('.')[0]
    vid_pred = Meitu.detect_video(vid,frames_dir)
    pred_dict[prefix] = vid_pred   
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,len(video_list),float(cnt_time) / (i+1)))
    if vid_pred == vid.label:
        correct += 1
    total += 1
total_time = time.time() - proc_start_time
print('total time:{:.02f}'.format(total_time))
print('Average class accuracy:{:.02f}%'.format(correct/float(total)*100))
with open('results/'+args.save_scores+'.txt','w') as fs:
    for key in pred_dict:
        fs.write(str(key) + ' '+ str(pred_dict[key]) + '\n')
    fs.close()
