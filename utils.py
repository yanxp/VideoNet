import torch
import collections
import torch.nn as nn
from torch.nn.init import normal, constant
def finetune(state_dict,model,num_class):
    modelname = list(model.state_dict().keys())
    new_params = collections.OrderedDict()
    new_fc = nn.Linear(2048, num_class)
    std = 0.001
    normal(new_fc.weight, 0, std)
    constant(new_fc.bias, 0)
    for i,layer_key in enumerate(list(state_dict.keys())):
        new_params[modelname[i]] = state_dict[layer_key]
        if modelname[i] == 'new_fc.weight':
            new_params[modelname[i]] = new_fc.weight
        if modelname[i] == 'new_fc.bias':
            new_params[modelname[i]] = new_fc.bias
    return new_params
def load_pretrained(model,path):
    target_weights=torch.load(path)
    own_state=model.state_dict()

    for name, param in target_weights.items():
        if name in 'features.18.weight' or name in 'features.18.bias':
            continue 
        if name in own_state:
            if isinstance(param,nn.Parameter):
                param=param.data
            try:
                if len(param.size())==5 and param.size()[3] in [3,7]:
                    own_state[name][:,:,0,:,:]=torch.mean(param,2)
                else:
                    own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}.\
                                   whose dimensions in the model are {} and \
                                   whose dimensions in the checkpoint are {}.\
                                   '.format(name,own_state[name].size(),param.size()))
        else:
           print ('{} meets error in locating parameters'.format(name))
    missing=set(own_state.keys())-set(target_weights.keys())

    print ('{} keys are not holded in target checkpoints'.format(len(missing)))

    return own_state
