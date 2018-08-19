import torch
import collections
import torch.nn as nn
from torch.nn.init import normal, constant
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self,num_class,alpha=None,gamma=2,size_average=True):
        super(FocalLoss,self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(num_class, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.num_class = num_class
        self.size_average = size_average
 
    def forward(self,inputs,target):
        b = inputs.size(0)
        c = inputs.size(1)
        p = F.softmax(inputs)
        class_mask = inputs.data.new(b, c).fill_(0)
        class_mask = Variable(class_mask)
        ids = target.view(-1,1)
        
        class_mask.scatter_(1, ids, 1.)
        
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (p*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

def finetune(modelpath,num_class):
    with open('layernames.txt','r') as f:
         names = [line.strip() for line in f.readlines()]
    new_params = collections.OrderedDict()
    kinetics_rgb = torch.load(modelpath)
    new_fc = nn.Linear(1024, num_class)
    std = 0.001
    normal(new_fc.weight, 0, std)
    constant(new_fc.bias, 0)
    for i,layer_key in enumerate(list(kinetics_rgb.keys())):
        new_params[names[i]] = kinetics_rgb[layer_key]
        if names[i] == 'module.new_fc.weight':
            new_params[names[i]] = new_fc.weight
        if names[i] == 'module.new_fc.bias':
            new_params[names[i]] = new_fc.bias
    return new_params

def loadMobileNetV2(state_dict,model,num_class):
    modelname = list(model.state_dict().keys())
    new_params = collections.OrderedDict()
    new_fc = nn.Linear(1280, num_class)
    std = 0.001
    normal(new_fc.weight, 0, std)
    constant(new_fc.bias, 0)
    for i,layer_key in enumerate(list(state_dict.keys())):
        new_params[modelname[i]] = state_dict[layer_key]
        if modelname[i] == 'module.new_fc.weight':
            new_params[modelname[i]] = new_fc.weight
        if modelname[i] == 'module.new_fc.bias':
            new_params[modelname[i]] = new_fc.bias
    return new_params

def loadShuffleNetV2(state_dict,model,num_class):
    modelname = list(model.state_dict().keys())
    new_params = collections.OrderedDict()
    new_conv = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
    for i,layer_key in enumerate(list(state_dict.keys())):
        
        new_params[modelname[i]] = state_dict[layer_key]
        if layer_key == 'network.8.weight':
            new_params[modelname[i]] = new_conv.weight
        if layer_key == 'network.8.bias':
            new_params[modelname[i]] = new_conv.bias
    return new_params

def loadShuffleNet(state_dict,model,num_class):
    modelname = list(model.state_dict().keys())
    new_params = collections.OrderedDict()
    new_fc = nn.Linear(1536, num_class)
    std = 0.001
    normal(new_fc.weight, 0, std)
    constant(new_fc.bias, 0)
    for i,layer_key in enumerate(list(state_dict.keys())):
        print(layer_key)
        new_params[modelname[i]] = state_dict[layer_key]
        if modelname[i] == 'module.new_fc.weight':
            new_params[modelname[i]] = new_fc.weight
        if modelname[i] == 'module.new_fc.bias':
            new_params[modelname[i]] = new_fc.bias
    return new_params

