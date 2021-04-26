import numpy as np
import torch
import torch.nn as nn

def mlp(input_shape,output_shape,n_layer):
    net = []
    for i in range(n_layer):
        if i==0:
            net.append(nn.Linear(in_features=input_shape,out_features=output_shape))
            net.append(nn.Tanh())
        else:
            net.append(nn.Linear(in_features=output_shape, out_features=output_shape))
            net.append(nn.Tanh())
    return nn.Sequential(*net)