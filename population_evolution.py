# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:50:16 2022

@author: darol
"""

from individual_ga_fede_evomutation_v2 import Individual

import numpy as np
import torch
import copy

class Population:
    
    def __init__(self, population_size=100, tot_generations=500): #,
                 # p_mutations=1., p_pruning_change=0.01,
                 # p_connect_max=0.1, p_disconnect_max=0.1):
        self._population_size = population_size
        self._population_size_half = int(self._population_size/2)
        # self._parents = parents
        self._tot_generations = tot_generations
        
        self._population_vector = np.zeros(population_size, dtype=object)
        self._fitness = np.zeros(population_size)
    
    
    def initialize_population(self, individual_best: Individual,
                              p_mutation_probability_pruning = 0.01,
                              p_mutations = 1.,
                              mutation_mean = 0.,
                              mutation_std = 0.001):
        
        self._population_vector[0] = individual_best        
        for i in range(1, self._population_size):
            offspring = copy.deepcopy(individual_best)
            '''mutate pruning operators hyperparams'''
            # if torch.rand(1) < p_mutation_probability_pruning:
            offspring.mutation_probability_pruning()
            
            offspring.mutation_pruning_genotypic(disconnect_forward=True)
            offspring.mutation_gaussian(p_mutations, 
                                        mutation_mean, 
                                        mutation_std)
            
            self._population_vector[i] = offspring
                          
        
    def set_fitness(self, individual_id, fitness):
        self._fitness[individual_id] = fitness
        
    
    def get_fitness_vector(self):
        return self._fitness
        
    
    def get_population_vector(self):
        return self._population_vector
        
    
    def reproduce(self,
                  p_mutation_probability_pruning = 0.01,
                  p_mutations = 1.,
                  mutation_mean = 0.,
                  mutation_std = 0.001):
        
        ranking = np.argsort(self._fitness)
        # return ranking[self._population_size_half:], ranking[:self._population_size_half]
        '''sort for memory contiguity, we do not care about order'''
        best = ranking[self._population_size_half:]
        best = best[np.argsort(best)]
        worst = ranking[:self._population_size_half]
        worst = worst[np.argsort(worst)]
        
        '''select and mutate'''        
        for i in range(self._population_size_half):
            offspring = copy.deepcopy(self._population_vector[best[i]])
            if torch.rand(1) < p_mutation_probability_pruning:
                offspring.mutation_probability_pruning()
            
            offspring.mutation_pruning_genotypic(disconnect_forward=True)
            offspring.mutation_gaussian(p_mutations, 
                                        mutation_mean, 
                                        mutation_std)
            self._population_vector[worst[i]] = copy.deepcopy(offspring)
            
    
    def get_individual(self, idx):
        return self._population_vector[idx]
            
    
    # def update_population_vector(self):
        
            
        
    
    
        
    
    
    
    
    