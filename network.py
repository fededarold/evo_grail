# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:26:29 2023

@author: darol
"""
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters 

import random
import copy


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

GRADIENT_FREE_LEARNING = True

#turn of autograd globally - do it if using gradient free methods
# torch.autograd.set_grad_enabled(False)

'''For now we use a fixed number of hidden layers, i.e. 2'''
class MLP(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size=40):
        super(MLP, self).__init__()
        
        # torch.autograd.set_grad_enabled(False)
        
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.tanh(x)
        
        return x
    
    def get_parameters_number(self):
        n_parameters = 0
        for i in self.parameters():
            n_parameters += i.numel()
            
        return n_parameters
    
    def set_parameters(self, params: torch.Tensor):
        vector_to_parameters(params, self.parameters())



class Agent():
    '''ATM we use only one hyperparameter'''
    def __init__(self, genome_parameters_size: int, initialization_hyperparameter: bool, 
                 num_experts: int, expert_population = 40): #genome_hyperparameters_range: list, genome_hyperparameters_size=1):
        self.genome_parameters_size = genome_parameters_size
        if initialization_hyperparameter:
            self._initialize_parameters_uniform()
        # else:
        self._initialize_parameters_normal(num_experts=num_experts)
     
    
    '''ATM we set only the probability of selecting the extrinsic goal'''    
    def _initialize_parameters_uniform(self):
        self.genome_hyperparameter = random.uniform(0., 1.)
        
                
    def _initialize_parameters_normal(self, num_experts: int, mean=0., std=0.01):
        self.genome_parameters = torch.normal(mean, std, 
                                              size=(num_experts, self.genome_parameters_size))
    
    
    def mutate(self, expert_id: int, std=0.001):
        self.genome_parameters[expert_id] += torch.normal(0, std, 
                                                          size=(1, self.genome_parameters_size))[0]
        

class NeuroEvolution():
    def __init__(self, population_size=20):
        self._population_size = population_size
        self._population = np.zeros(population_size, dtype=object)
        self._fitness = np.zeros(population_size)
    
    
    def initialize_population(self, genome_parameters_size: int, initialization_hyperparameter: bool, 
                              num_experts: int, homogeneous_genomes=False):
        for i in range(self._population_size):
            if not homogeneous_genomes:
                a = Agent(genome_parameters_size, initialization_hyperparameter, num_experts)
                self._population[i] = a
            else: 
                if i==0:
                    a = Agent(genome_parameters_size, initialization_hyperparameter, num_experts)
                self._population[i] = copy.deepcopy(a)
    
        
    def selection_rank(self, n_parents: int, expert_id: int):
        rank = self._fitness.argsort()[::-1]
        best_agents = copy.deepcopy(self._population[rank[:n_parents]])        
        for a in best_agents:
            a.mutate(expert_id)            
        self._population[rank[n_parents:]] = copy.deepcopy(best_agents)
        self._reset_fitness()
        
    
    def _reset_fitness(self):
        self._fitness *= 0
        
    
    def set_fitness(self, agent_id: int, fitness):
        self._fitness[agent_id] = fitness
        
        
mlp = MLP(4,2,hidden_size=3)
#Disable autograd if we use gradient free methods, e.g. neuroevolution
if GRADIENT_FREE_LEARNING:
    for params in mlp.parameters():
        params.requires_grad=False
        

neuro = NeuroEvolution()
neuro.initialize_population(genome_parameters_size=mlp.get_parameters_number(),
                            initialization_hyperparameter=True, 
                            num_experts=3)

for i in range(neuro._population_size):
    neuro.set_fitness(i, np.random.randint(100))
    
neuro.selection_rank(10, 0)

agent = Agent(3, "uniform", 1)

# agent_test = Agent(mlp.get_parameters_number(), genome_hyperparameters_size=0, 
#                    genome_hyperparameters_range=[None], num_experts=1)
# print("genome")
# print(agent_test.genome_parameters)

# print("model")
# mlp.set_parameters(agent_test.genome_parameters[0])
# print(parameters_to_vector(mlp.parameters()))

# print("genome mutated")
# agent_test.mutate(id_expert=0)
# print(agent_test.genome_parameters)

# print("model mutated")
# mlp.set_parameters(agent_test.genome_parameters[0])
# print(parameters_to_vector(mlp.parameters()))

# x = torch.rand(4)
# y = mlp.forward(x)