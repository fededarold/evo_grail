# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:13:58 2021

@author: Fede Laptop
"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, mask_list):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.tanh = nn.Tanh()
        self.hidden = []
        for i in range(1, len(hidden_sizes)):
            self.hidden.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.fc2 = nn.Linear(hidden_sizes[-1], output_size)
        #self.mask = mask
        '''binary masks input.'''
        self.mask=[]
        if mask_list[0] is not False:
            self.mask.append(torch.zeros((hidden_sizes[0], input_size), dtype=torch.bool))
        else:
            self.mask.append(False)
        '''binary masks hidden. we do not do last hidden to output for now'''
        for i in range(1, len(mask_list)-1):
            if mask_list[i] is not False:
                self.mask.append(torch.zeros((hidden_sizes[i], hidden_sizes[i-1]), dtype=torch.bool))
            else:
                self.mask.append(False)
                
        '''we do not do last hidden to output for now'''
        if mask_list[-1] is not False:
            self.mask.append(torch.zeros((output_size, hidden_sizes[-1]), dtype=torch.bool))
        
        '''we need to align the genome as we use a list for hidden layers'''
        self.genome_trimmer = []
        self.total_parameters_weights = 0
        self.total_parameters_masks = 0
        self.parameters_count()
    
                
    def forward(self, x):
        x = torch.tensor(x)
        # if self.mask[0] is not False:
        #     pruned_weights = torch.mul(out, self.mask[0])
        #out = self.fc1(x)
        if self.mask[0] is not False:
            pruned_weights = torch.mul(self.fc1.weight, self.mask[0])
            out = torch.matmul(pruned_weights, x)
            # print(out)
            out = torch.add(out, self.fc1.bias)
        else:
            out = self.fc1(x)
        out = self.tanh(out)
        '''binary masks. if there is a tensor element-wise multiplication'''
        # if self.mask[0] is not False:
        #     out = torch.mul(out, self.mask[0])
        
        for i, hidden in enumerate(self.hidden):            
            if self.mask[i+1] is not False:
                pruned_weights = torch.mul(hidden.weight, self.mask[i+1])
                out = torch.matmul(pruned_weights, out)
                # print(out)
                #out = torch.matmul(hidden.weight, out)
                out = torch.add(out, hidden.bias)
            else:
                out = hidden(out)
            '''binary masks. if there is a tensor element-wise multiplication'''
            # if self.mask[i+1] is not False:
            #     out = torch.mul(out, self.mask[i+1])
            out = self.tanh(out)
        
            
        '''output and mask'''
        if self.mask[-1] is not False:
            pruned_weights = torch.mul(self.fc2.weight, self.mask[-1])
            out = torch.matmul(pruned_weights, out)
            out = torch.add(out, self.fc2.bias)
        else:
            out = self.fc2(out)
        out = self.tanh(out)
        
        #return out
        return out.detach().numpy()
    
    
    '''we do an init for the model and we count paramters just once, no need to do it all the time
    we write a getter to initiate the individual'''

    def parameters_count(self):
        
        '''WEIGHTS FIRST'''
        '''input to hidden and last hidden to output is here -> i.e. fc1 and fc2'''
        params = len(parameters_to_vector(self.parameters()))
        self.genome_trimmer.append(params)
        '''we now do the hidden'''
        for i in range(len(self.hidden)):
            params_hidden = len(parameters_to_vector(self.hidden[i].parameters()))
            self.genome_trimmer.append(params_hidden)
            params += params_hidden
        self.total_parameters_weights = params
        
        '''now MASKS'''
        params = []
        for m in self.mask:
            if m is not False:
                params.append(m.size())
        self.total_parameters_masks = params
            
        
    def param_weights_count(self):
        return self.total_parameters_weights
        # '''input to hidden and last hidden to output is here -> i.e. fc1 and fc2'''
        # params = parameters_to_vector(self.parameters())
        # self.genome_trimmer.append(params)
        # '''we now do the hidden'''
        # for i in range(self.hidden):
        #     params_hidden = parameters_to_vector(self.hidden[i])
        #     self.genome_trimmer.append(params_hidden)
        #     params += params_hidden
        # return len(params)
    
        
    # def param_masks_count(self):
    #     params = 0
    #     for m in self.mask:
    #         if m is not False:
    #             params += len(parameters_to_vector(m))
    #     return params
    
    def param_masks_count(self):
        return self.total_parameters_masks
        # params = []
        # for m in self.mask:
        #     if m is not False:
        #         params.append(m.size())
        # return params

    '''MUST BE TORCH TENSOR'''
    def set_model_params(self, weights: torch.tensor, masks: torch.tensor):
        
        '''map fc1 and fc2, i.e. input and output weights'''
        vector_to_parameters(weights[0:self.genome_trimmer[0]], self.parameters())
        for i in range(len(self.genome_trimmer)-1):
            vector_to_parameters(weights[
                self.genome_trimmer[i]:self.genome_trimmer[i]+self.genome_trimmer[i+1]], 
                self.hidden[i].parameters())
        
        '''deep copy equivalent (sort of, detach from graph)'''
        j=0
        for i in range(len(self.mask)):
            if self.mask[i] is not False:
                self.mask[i] = masks[j].clone().detach()
                j += 1
    
        #print(f"DEBUG set_model_params duration: {timer() - start_time}")
 
    
# test = MLP(4, [5, 5], 3, [True, False])