# -*- coding: utf-8 -*-

import os,sys
import random
import cv2
import torch
from PIL import Image
from torch.autograd import Variable

MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(MY_DIRNAME)
from defines import *
from models import TSN
from transforms import *
from basic_ops import *
from PIL import Image


class ServerApi(object):
    """
    统一算法预测接口类：
    注：
        1.handle为举办方验证接口，该接口必须返回预测分类值，参赛队伍需具体实现该接口
        2.模型装载操作必须在初始化方法中进行
        3.初始化方法必须提供gpu_id参数
        3.其他接口都为参考，可以选择实现或删除
    """
    def __init__(self, gpu_id=0):
        self.args = parser.parse_args()
        self.model = self.load_model(gpu_id)

    def video_frames(self, file_dir):
        """
        视频截帧
        :param file_dir: 视频路径
        :return:
        """
        videoCapture = cv2.VideoCapture(file_dir)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        images = list()
        for i in range(int(frames)):
            ret,frame = videoCapture.read()
            if not ret:
                continue
            images.append(frame)
        offsets = self.get_test_indices(len(images))
        return self.get(offsets,images)

    def get_test_indices(self,num_frames):
        '''
        return the frame index
        '''
        tick = (num_frames - self.args.new_length + 1 ) /float(self.args.test_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.args.test_segments)])
        return offsets

    def get(self,offsets,frames):
        '''
        return the indexed images
        '''
        images = list()
        for ind in offsets:
            p = int(ind)
            img = frames[p]
            images.extend([Image.fromarray(img).convert('RGB')])
        
        return images


    def load_model(self, gpu_id):
        """
        模型装载
        :param gpu_id: 装载GPU编号
        :return:
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        net = TSN(self.args.num_class, 1, self.args.modality,base_model=self.args.arch,consensus_type=self.args.crop_fusion_type,dropout=self.args.dropout) 
        checkpoint = torch.load(self.args.weights)         
        base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
        net.load_state_dict(base_dict)
        cropping = torchvision.transforms.Compose([GroupScale(net.scale_size), GroupCenterCrop(net.input_size),])
        self.transform=torchvision.transforms.Compose([cropping,Stack(roll=self.args.arch == 'BNInception'),ToTorchFormatTensor(div=self.args.arch != 'BNInception'),GroupNormalize(net.input_mean, net.input_std)])
        net = net.cuda()
        net.eval()
        return net

    def predict(self, file_dir):
        """
        模型预测
        :param file_dir: 预测文件路径
        :return:
        """
        frames = self.video_frames(file_dir)
        input = self.transform(frames)
        input = input.unsqueeze(0)
        input_var = Variable(input.view(-1, 3, input.size(2), input.size(3)),volatile=True).cuda()
        rst = self.model(input_var).data.cpu().numpy().copy()
        vid_pred = rst.reshape((self.args.test_crops, self.args.test_segments, self.args.num_class)).mean(axis=0).reshape((self.args.test_segments, 1, self.args.num_class))
        pred_class = np.argmax(np.mean(vid_pred, axis=0))
        return pred_class

    def handle(self, video_dir):
        """
        算法处理
        :param video_dir: 待处理单视频路径
        :return: 返回预测分类
        """
        return self.predict(video_dir)
