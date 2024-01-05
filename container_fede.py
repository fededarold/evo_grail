# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:53:25 2021

@author: Fede Laptop
"""

import numpy as np
import torch
from individual_ga_fede import Individual
from MLP_fede import MLP

import random
import copy


# class Container:
#     def __init__(self, feature_x_size, feature_y_size, initial_population, budget, batch, selection_probability):
#         self.grid = np.empty((feature_x_size, feature_y_size), dtype=object)

class Container:
    def __init__(self, grid_x_size: int, grid_y_size: int, 
                 neural_network: MLP, 
                 #beta_distribution_alpha: float, beta_distribution_beta: float, init_fully_connected=False, 
                 feature_x_min=0., feature_x_max=1., 
                 feature_y_min=0., feature_y_max=1.):
        #self.init_fully_connected = init_fully_connected
        self.grid = np.empty((grid_x_size, grid_y_size), dtype=object)
        self.x = (feature_x_max - feature_x_min) / self.grid.shape[0]
        self.y = (feature_y_max - feature_y_min) / self.grid.shape[1]
        self.grid_filled_id = None
        self.disconnectable_units = 0
        for i in range(len(neural_network.mask)):
            if neural_network.mask[i] is not False:
                self.disconnectable_units += neural_network.mask[i].shape[0]
        #self.init_container(initial_population=10, neural_network=mlp_debug)
    
    
    '''place the guy in the right cell'''
    #def place_in_cell(self, x, y, individual: Individual):
    def place_individual_in_cell(self, individual: Individual, grid):
        idx = int(np.floor(individual.get_feature_sparsity() / self.x ))        
        idy = int(np.floor(individual.get_feature_disconnected_units(self.disconnectable_units) / self.y ))
        if idx == grid.shape[0]:
            idx -= 1
        if idy == grid.shape[1]:
            idy -= 1
        
        # print(idx, '  ', idy) 
        # print(individual.get_feature_sparsity(), '  ', individual.get_feature_disconnected_units(self.disconnectable_units)) 
        
        if grid[idx][idy] is not None:
            
            new = individual.get_fitness()
            old = grid[idx][idy].get_fitness()
            
            #if individual.get_fitness() > self.grid[idx][idy].get_fitness():
            if new > old:
                grid[idx][idy] = copy.deepcopy(individual)
            #     print("replacing  ", [idx, idy])
            # else:
            #     print('looser  ', [idx, idy])
        else:
            grid[idx][idy] = copy.deepcopy(individual)
            # self.grid_filled_id.append([idx, idy])
            # print('populating:  ', [idx, idy])
            #if [idx, idy] not in self.grid_filled_id:
            #    self.grid_filled_id.append([idx, idy])
                #print('not in list  ',  [idx, idy])
           # else:
                #print('in list  ',  [idx, idy])
        return grid
    

    def select_random_individual(self, number_of_individuals=1):
        return np.random.choice(len(self.grid_filled_id[0]), number_of_individuals)
        # individual_id = np.random.choice(len(self.grid_filled_id[0]))
        # return copy.deepcopy(self.grid[self.grid_filled_id[0][individual_id]][self.grid_filled_id[1][individual_id]]), [self.grid_filled_id[0][individual_id], self.grid_filled_id[1][individual_id]]  
    
    def select_roulette_individual(self, number_of_individuals=1):
        # fitness_all = []
        #grid_filled_id_ordered = []
        # for i in range(self.grid.shape[0]):
        #     for j in range(self.grid.shape[1]):
        #         if self.grid[i][j] is not None:
        #             fitness_all.append(self.grid[i][j].get_fitness())
        #             #grid_filled_id_ordered.append([i, j])
        
        fitness_all = np.zeros(len(self.grid_filled_id[0]))
        for i in range(len(self.grid_filled_id[0])):
            fitness_all[i] = self.grid[self.grid_filled_id[0][i]][self.grid_filled_id[1][i]].get_fitness()
        '''set min to 0'''
        # fitness_min = min(fitness_all)
        fitness_min = np.min(fitness_all)
        if fitness_min < 0:
            fitness_min *= -1
        fitness_all += fitness_min
            # for i in range(len(fitness_all)):
            #     fitness_all[i] += fitness_min
        '''create p of being selected'''
        '''if only one cell of the grid set p=1'''
        if len(fitness_all) == 1:
            fitness_all[0] = 1.
        else:
            # fitness_sum = sum(fitness_all)
            # for i in range(len(fitness_all)):
            #     fitness_all[i] /= fitness_sum
            fitness_all /= np.sum(fitness_all)
        ''' select'''
        return np.random.choice(len(self.grid_filled_id[0]), 
                                number_of_individuals.item(), 
                                p=fitness_all)
        #individual_id = np.random.choice(len(grid_filled_id_ordered), p=fitness_all)
        # individual_id = np.random.choice(len(self.grid_filled_id[0]), p=fitness_all)
        #return copy.deepcopy(self.grid[self.grid_filled_id[0][individual_id]][self.grid_filled_id[1][individual_id]]), [self.grid_filled_id[0][individual_id], self.grid_filled_id[1][individual_id]] 
    
    def get_individual(self, individual_index):
        return copy.deepcopy(self.grid[self.grid_filled_id[0][individual_index]][self.grid_filled_id[1][individual_index]])
        
    def get_grid(self, deepcopy=False):
        if deepcopy:
            return copy.deepcopy(self.grid)
        else:
            return self.grid
        
    
    def set_grid(self, new_grid):
        self.grid = copy.deepcopy(new_grid)
    
    
    def update_grid_occupancy(self):
        self.grid_filled_id = np.where(self.grid != None)
    
        

# random.seed(1)
# torch.manual_seed(1)
# np.random.seed(1)
        
# mlp_debug = MLP(5, [4, 3], 2, [True, True])
# container = Container(10, 10, 20, mlp_debug, False)
# grid = container.get_grid()  