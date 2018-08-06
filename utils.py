import torch
import collections
import torch.nn as nn
from torch.nn.init import normal, constant
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

def loadShuffleNet(state_dict,model,num_class):
    modelname = list(model.state_dict().keys())
    new_params = collections.OrderedDict()
    new_fc = nn.Linear(1536, num_class)
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

