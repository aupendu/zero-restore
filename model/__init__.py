import os
import torch
import torch.nn as nn
from importlib import import_module

class Model(nn.Module):
    def __init__(self, modelname):
        super(Model, self).__init__()
        
        print('Making model...')
        
        module = import_module('model.' + modelname)
        self.model = module.make_model()

    def forward(self, x):
        return self.model(x)
