# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:36:26 2023

@author: darol
"""

import torch
import numpy as np
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes=None, 
                 activation_function=nn.ReLU(), activation_output=nn.Tanh(),
                 optimizer=None):
        super(MLP, self).__init__()
        
        if hidden_sizes is None:
            #perceptron
            hidden_sizes=[input_size, output_size]
        else:
            # handle 1 layer passed as integer
            if isinstance(hidden_sizes, int):
                hidden_sizes = [hidden_sizes]
            #MLP n layers
            hidden_sizes.insert(0, input_size)
            hidden_sizes.append(output_size)
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes)-1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
        
        self.activation_function = activation_function             
        self.activation_output = activation_output
        
        self.optimizer=optimizer
        
        
    def forward(self, x):
        
        y = x
        for i in range(len(self.layers)-1):
            y = self.layers[i](y)
            y = self.activation_function(y)
        y = self.layers[-1](y)
        y = self.activation_output(y)
        return y
        
    

if __name__ == "__main__":
    
    x = torch.Tensor([0.1, 0.2])
    
    perceptron = MLP(2,3)
    print(perceptron(x))
    
    perceptron_hidden = MLP(2,3,2)
    print(perceptron_hidden(x))
    
    perceptron_multi = MLP(2,3,[2, 3, 4])
    print(perceptron_multi(x))
    