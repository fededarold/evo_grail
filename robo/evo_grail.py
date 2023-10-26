# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:26:29 2023

@author: darol
"""
import torch
import numpy as np
from torch.nn.utils import vector_to_parameters 

import random

from neuroevolution import NeuroEvolution, AgentBase



#turn of autograd globally - do it if using gradient free methods
# torch.autograd.set_grad_enabled(False)

'''For now we use a fixed number of hidden layers, i.e. 2'''

# TODO: pass function instead of string. e.g., torch.nn.ReLU()
class MLP(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size=40, 
                 activation_function=torch.nn.ReLU(), 
                 activation_output=torch.nn.Tanh()):
        super(MLP, self).__init__()
        
        # self.debug_id = debug_id
        
        torch.autograd.set_grad_enabled(False)
        
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        
        
        self.activation_function = activation_function             
        self.activation_output = activation_output
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_function(x)
        x = self.fc2(x)
        x = self.activation_function(x)
        x = self.fc3(x)
        x = self.activation_output(x)
        
        return x
    
    def get_parameters_number(self):
        n_parameters = 0
        for i in self.parameters():
            n_parameters += i.numel()
            
        return n_parameters
    
    def set_parameters(self, params: torch.Tensor):
        vector_to_parameters(params, self.parameters())


class AgentBandit(AgentBase):
    
    def __init__(self, n_goals: int, neural_net: MLP,
                 bandit_time_horizon=10, **experts_parameters):
        super(AgentBandit, self).__init__(genome_size=1, genome_data_type="python",
                                          random_distribution="uniform")
        self._bandit_time_horizon = bandit_time_horizon
        self._n_goals=n_goals
        # self._experts_parameters = experts_parameters["experts_parameters"]
        self.neural_net = neural_net
        
        agent_expert = AgentBase(genome_size=neural_net.get_parameters_number())
        self._experts = []
        for i in range(self._n_goals):
            self._experts.append(NeuroEvolution(population_size=4, 
                                                agent=agent_expert, 
                                                randomizer_parameters=experts_parameters["experts_parameters"]))
            

if __name__ == "__main__":
    experts_parameters = {"random initialization": {"mean": 0.,
                                                    "std": 0.1},
                          "mutations": {"mean": 0.,
                                        "std": 0.001}}       
            
        
    bandit_parameters = {"random initialization": {"lower bound": 0.,
                                                   "upper bound": 1.},
                         "mutations": {"lower bound": -0.01,
                                       "upper bound": 0.01}}
    
    
    nn = MLP(input_size=3, output_size=1, hidden_size=2,
             debug_id=1)
    
    
    agent_bandit = AgentBandit(n_goals=3, neural_net=nn, 
                               experts_parameters=experts_parameters)
    
    # a = AgentBandit(bandit_time_horizon=10, 
    #                 genome_data_type="python", 
    #                 random_distribution="uniform")
    
    n = NeuroEvolution(population_size=5, agent=agent_bandit, 
                       randomizer_parameters=bandit_parameters)
    
