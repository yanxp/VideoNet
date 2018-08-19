import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
import pickle as pkl
import os

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics','meitu'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=6)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
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

#中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
    def forward(self, x):
        return self.submodule.base_model(x)

net = TSN(num_class, 1, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,before_softmax=True,
          dropout=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}

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

data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.test_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

net=FeatureExtractor(net)
net = torch.nn.DataParallel(net)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
print('total_num:{}'.format(total_num))
output = []
video_name = []

def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)

    return net(input_var).data.cpu().numpy().copy()

   # rst = net(input_var).data.cpu().numpy().copy()
   # return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
   #     (args.test_segments, 1, num_class)
   # ), label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, (data,label,name) in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data, label))
    output.append(rst)
    video_name.append(name)
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))
#train_data = {'name':video_name,'feature':output}
for i,name in enumerate(video_name):
    pkl.dump(output[i], open(os.path.join('train_features',name[0]+'.pkl'), "wb"))

'''
video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
fs = open('category_val_before_softmax.txt','w')
for i,x in enumerate(output):
    out = np.mean(x[0], axis=0)[0]
    #out = softmax(out)
    fs.write(video_name[i][0])
    for prob in out:
        fs.write(',{:.04f}'.format(prob))
    fs.write('\n')    
fs.close()
with open('results/'+args.save_scores+'.txt','w') as fs:
    for i in range(len(video_pred)):
        fs.write(str(video_name[i][0])+' '+str(video_pred[i])+'\n')
    fs.close()


video_labels = [x[1] for x in output]

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

import collections
d = collections.defaultdict(list)
fs = open('data/meitu_splits/classind.txt','r')
classind = [line.strip().split(',') for line in fs.readlines()]
for i in range(len(classind)):
    d[i] = classind[i][1]
print('per class accuracy===>')
for i in range(len(cls_acc)):
    print('{} {:.02f}'.format(d[i],cls_acc[i]))

print('Average accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))


correct = 0
total = 0
with open('results/'+args.save_scores+'_wrong.txt','w') as fs:
    for i in range(len(video_pred)):
        if video_labels[i] != video_pred[i]:
            fs.write(str(video_name[i][0])+',True:'+d[int(video_labels[i])]+ ',False:'+ d[int(video_pred[i])]+'\n')
        else:
            correct += 1
        total += 1
    fs.close()
print('correct / total accuracy {:.02f}%'.format(correct/total*100))

with open('results/'+args.save_scores+'_per_class.txt','w') as fs:
    fs.write('average second / video:{} \n'.format(averagetime) )
    for i in range(len(cls_acc)):
        fs.write('{} {:.02f} \n'.format(d[i],cls_acc[i]))
    fs.write('Average accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    fs.close()
'''
