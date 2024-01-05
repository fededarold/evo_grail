# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:45:26 2021

@author: Fede Laptop
"""

import torch
import numpy as np
from MLP_fede import MLP

'''to avoid mistakes we represent the weights as a vector (built in torch function) and masks as bidimensional tensors
we use a beta distribution to sample the threshold for mask init.. this way we can decide a centroid where the 
initial population will be placed in the grid'''
class Individual:    
    #def __init__(self, genome_weights_length, genome_masks_length: [], init_fully_connected: bool):
    def __init__(self, neural_network: MLP, 
                 beta_distribution_alpha: float, beta_distribution_beta: float, 
                 init_fully_connected: bool, 
                 individual_id: int
                 ):
        self.fitness = 0
        self.individual_id = individual_id
        
        ''' generation - sparsity - disconnection'''
        # self.evolutionary_history = [] 
        
        genome_weights_length = neural_network.param_weights_count()
        genome_masks_length = neural_network.param_masks_count()
        ''' TODO: 1) initialiazation and 2) check dtype'''
        self.genome_weights = torch.randn(genome_weights_length)
        '''these are bidimesional'''
        self.genome_masks = []
        '''p_connect and disconnect'''
        self.p_connect = torch.rand(1)
        self.p_disconnect = torch.rand(1)
        
        '''initialize a beta sampler'''
        beta_sampler = torch.distributions.beta.Beta(beta_distribution_alpha, beta_distribution_beta)
        for i in range(len(genome_masks_length)):
            if init_fully_connected:
                self.genome_masks.append(torch.ones(genome_masks_length[i], dtype=torch.bool))
            else:
                '''trick to differentiate connectivity among the population to equally ditribute it'''
                #p_disconnect = torch.rand(1)
                p_disconnect = beta_sampler.sample()
                p_mask = torch.rand(genome_masks_length[i])
                mask = p_mask > p_disconnect
                self.genome_masks.append(mask)
                #self.genome_masks.append(torch.randint(2, tuple(genome_masks_length[i]), dtype=torch.bool))
        self.alignment = self.map_masks_to_weights(neural_network)
    
    
    '''very messy we need to re-think this'''            
    def map_masks_to_weights(self, neural_network: MLP):
        #alignment = []
        alignment = torch.Tensor()
        if torch.is_tensor(neural_network.mask[0]):
            #alignment.append((0, torch.numel(neural_network.fc1.weights)))
            alignment = torch.cat((alignment, torch.arange(0, torch.numel(neural_network.fc1.weight))), 0)
        params = torch.numel(neural_network.fc1.weight) + torch.numel(neural_network.fc1.bias)
        
        if torch.is_tensor(neural_network.mask[-1]):
            #alignment.append((0, torch.numel(neural_network.fc1.weights)))
            alignment = torch.cat((alignment, torch.arange(params, params + torch.numel(neural_network.fc2.weight))), 0)
        params += torch.numel(neural_network.fc2.weight) + torch.numel(neural_network.fc2.bias)
                              
        for i in range(1, len(neural_network.mask)-1):
            if torch.is_tensor(neural_network.mask[i]):
                #alignment.append((params, neural_network.hidden[i].weight))
                alignment = torch.cat((alignment, torch.arange(params, params + torch.numel(neural_network.hidden[i-1].weight))), 0)
            params += torch.numel(neural_network.hidden[i-1].weight) + torch.numel(neural_network.hidden[i-1].bias)
        return alignment.int()
    
    
    def get_feature_sparsity(self):
        if len(self.genome_masks) == 0:
            return 0
        
        tot_parameters = 0
        actual_parameters = 0
        for m in self.genome_masks:
            tot_parameters += torch.numel(m)
            actual_parameters += torch.sum(m) 
        return 1 - (actual_parameters / tot_parameters)
    
    
    '''we chack just incoming weights. If we cut biases we must consider this as a deleted connection forward as well'''
    '''precalculate the number of disconnectable units outside to avoid recomputing them every time'''
    def get_feature_disconnected_units(self, tot_disconnectable_units):
        if tot_disconnectable_units == 0:
            return tot_disconnectable_units
        
        disconnected_units = 0
        for m in self.genome_masks:
            connections = torch.sum(m, axis=0)                
            disconnected_units += torch.numel(connections) - torch.count_nonzero(connections)
        return disconnected_units / tot_disconnectable_units
    
    
    # def set_evolutionary_history(self, generation, tot_disconnectable_units):
    #     self.evolutionary_history.append((generation, self.individual_id, self.fitness, self.get_feature_sparsity(), self.get_feature_disconnected_units(tot_disconnectable_units)))
    
    
    # def mutation_polynomial_bounded(genome_weights: torch.tensor, mutation_probability: float, eta: float, lower: float, upper: float):
    #     for i in range(len(genome_weights)):
    #             if torch.rand(1) < mutation_probability:
    #                 x = genome_weights[i]
    #                 delta_1 = (x - lower) / (upper - lower)
    #                 delta_2 = (upper- x) / (upper - lower)
    #                 rand = torch.rand(1)
    #                 mut_pow = 1. / (eta + 1.)

    #                 if rand < 0.5:
    #                     xy = 1. - delta_1
    #                     val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
    #                     delta_q = val**mut_pow - 1.
    #                 else:
    #                     xy = 1. - delta_2
    #                     val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
    #                     delta_q = 1. - val**mut_pow

    #                 x += delta_q * (upper - lower)
    #                 x = min(max(x, lower), upper)
    #                 genome_weights[i] = x
    #     return genome_weights
    
    def mutation_polynomial_bounded(self, mutation_probability: float, eta: float, lower: float, upper: float):
        mask_flatten = torch.Tensor()
        for i in range(len(self.genome_masks)):
            if torch.is_tensor(self.genome_masks[i]):
                mask_flatten = torch.cat((mask_flatten, self.genome_masks[i].flatten()))
        
        mask_pointer = 0
        do_mutate = True   
        
        for i in range(len(self.genome_weights)):
            
            '''first check if we are in mask range'''
            is_in_mask = self.alignment == i
            if is_in_mask.any():                
                '''check if the weight is pruned'''
                if mask_flatten[mask_pointer]:
                    do_mutate = True
                else:
                    do_mutate = False
                mask_pointer += 1
            else:
                do_mutate = True  
            
            if torch.rand(1) < mutation_probability and do_mutate:
                                    
                x = self.genome_weights[i]
                delta_1 = (x - lower) / (upper - lower)
                delta_2 = (upper- x) / (upper - lower)
                rand = torch.rand(1)
                mut_pow = 1. / (eta + 1.)

                if rand < 0.5:
                    xy = 1. - delta_1
                    val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
                    '''Numpy does not handle fractional power of negative numbers'''
                    if val < 0:    
                        delta_q = - (np.abs(val)) ** mut_pow - 1.
                    else:
                        delta_q = val ** mut_pow - 1.
                        
                else:
                    xy = 1. - delta_2
                    val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
                    '''Numpy does not handle fractional power of negative numbers'''
                    if val < 0:
                        delta_q = 1. - (np.abs(val)) ** mut_pow
                    else:
                        delta_q = 1. - val**mut_pow
                
                x = x.unsqueeze(0)    
                x += delta_q * (upper - lower)
                x = min(max(x, lower), upper)
                
                '''prbably next line is pointless'''
                self.genome_weights[i] = x
                    
    
    def mutation_gaussian(self, mutation_probability: float,  mean: float, std: float):
        mask_flatten = torch.Tensor()
        for i in range(len(self.genome_masks)):
            if torch.is_tensor(self.genome_masks[i]):
                mask_flatten = torch.cat((mask_flatten, self.genome_masks[i].flatten()))
        
        mask_pointer = 0
        do_mutate = True        
        for i in range(len(self.genome_weights)):
            
            '''first check if we are in mask range'''
            is_in_mask = self.alignment == i
            if is_in_mask.any():                
                '''check if the weight is pruned'''
                if mask_flatten[mask_pointer]:
                    do_mutate = True
                else:
                    do_mutate = False
                mask_pointer += 1
            else:
                do_mutate = True               
                    
                
            if torch.rand(1) < mutation_probability and do_mutate:
                    
                x = self.genome_weights[i]
                x = x.unsqueeze(0)
                x += torch.empty(1).normal_(mean=mean,std=std)
                self.genome_weights[i] = x
               
        
    '''we drop out units without any incoming connection, thus pruning all its forward weights''' 
    def pruning_dropout_forward(self):
        for m in range(len(self.genome_masks)-1):
            if self.genome_masks[m] is not False:
                for row in range(self.genome_masks[m].shape[0]):
                    if torch.sum(self.genome_masks[m][row]) == 0:
                        self.genome_masks[m+1][:,row] = False
                    
        
        
        
    # def mutation_pruning(genome_masks: torch.tensor, disconnection_probability: float, connection_probability: float):
    #     for i in range(genome_masks.size()[0]):
    #         for j in range(genome_masks.size()[1]):
    #             if genome_masks[i][j] == True and torch.rand(1) < disconnection_probability:
    #                 genome_masks[i][j] = False
    #             elif genome_masks[i][j] == False and torch.rand(1) < connection_probability:
    #                 genome_masks[i][j] = True
    #     return genome_masks
    
    def mutation_pruning(self, disconnection_probability: float, connection_probability: float, disconnect_forward: bool):
        for m in range(len(self.genome_masks)):
            for i in range(self.genome_masks[m].shape[0]):
                for j in range(self.genome_masks[m].shape[1]):
                    if self.genome_masks[m][i][j] == True and torch.rand(1) < disconnection_probability:
                        self.genome_masks[m][i][j] = False
                    elif self.genome_masks[m][i][j] == False and torch.rand(1) < connection_probability:
                        self.genome_masks[m][i][j] = True
                        
        if disconnect_forward:
            self.pruning_dropout_forward()
        return self.genome_masks
    
    def mutation_pruning_genotypic(self, disconnect_forward: bool):
        for m in range(len(self.genome_masks)):
            for i in range(self.genome_masks[m].shape[0]):
                for j in range(self.genome_masks[m].shape[1]):
                    if self.genome_masks[m][i][j] == True and torch.rand(1) < self.p_disconnect:
                        self.genome_masks[m][i][j] = False
                    elif self.genome_masks[m][i][j] == False and torch.rand(1) < self.p_connect:
                        self.genome_masks[m][i][j] = True
                        
        if disconnect_forward:
            self.pruning_dropout_forward()
        return self.genome_masks 
    
    def mutation_probability_pruning(self):
        self.p_connect = torch.rand(1)
        self.p_disconnect = torch.rand(1)
    
    def get_fitness(self):
        return self.fitness
    
    
    def reset_fitness(self):
        self.fitness = 0
    
    def set_fitness(self, fitness):
        self.fitness += fitness
    
    
    def get_features(self):
       self.features["sparsity"] = self.get_feature_sparsity()
       self.features["disconnected"] = self.get_feature_disconnected_units()
       return self.features
       
                        
        
 # mutation_probability, 
 # disconnection_probability, connection_probability       
        
        # def mut_polynomial_bounded(individual: MutableSequence[Any], eta: float, low: float, up: float, mut_pb: float) -> MutableSequence[Any]:
        #     """Return a polynomial bounded mutation, as defined in the original NSGA-II paper by Deb et al.
        #     Mutations are applied directly on `individual`, which is then returned.
        #     Inspired from code from the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py).

        #     Parameters
        #     ----------
        #     :param individual
        #         The individual to mutate.
        #     :param eta: float
        #         Crowding degree of the mutation.
        #         A high ETA will produce mutants close to its parent,
        #         a small ETA will produce offspring with more differences.
        #     :param low: float
        #         Lower bound of the search domain.
        #     :param up: float
        #         Upper bound of the search domain.
        #     :param mut_pb: float
        #         The probability for each item of `individual` to be mutated.
        #     """
        #     for i in range(len(individual)):
        #         if random.random() < mut_pb:
        #             x = individual[i]
        #             delta_1 = (x - low) / (up - low)
        #             delta_2 = (up - x) / (up - low)
        #             rand = random.random()
        #             mut_pow = 1. / (eta + 1.)

        #             if rand < 0.5:
        #                 xy = 1. - delta_1
        #                 val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
        #                 delta_q = val**mut_pow - 1.
        #             else:
        #                 xy = 1. - delta_2
        #                 val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
        #                 delta_q = 1. - val**mut_pow

        #             x += delta_q * (up - low)
        #             x = min(max(x, low), up)
        #             individual[i] = x
        #     return individual
        