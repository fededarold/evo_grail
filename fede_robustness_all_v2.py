# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:43:26 2022

@author: darol
"""

from gymnasium.envs.box2d import BipedalWalker, BipedalWalkerHardcore, LunarLander, LunarLanderContinuous
import gymnasium as gym

from MLP_fede import MLP
from individual_ga_fede_evomutation_v2 import Individual
from container_fede import Container

import random
import torch
import numpy as np

class Robustness:
    
    def __init__(self, n_trials,
                 path, f_name, 
                 model, grid,
                 params, env):
        
        self.n_trials = n_trials
        
        self.path = path
        self.f_name = f_name
        
        self.model = model
        self.grid = grid
        
        self.params = params
        self.env = env
        
        self.robustness = np.zeros((self.n_trials, self.grid.shape[0], self.grid.shape[1]))
        self.robustness[:] = np.nan 
                   
    
    def run_test(self):
        
        '''test individual'''            
        ind = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] is not None:
                    
                    print(ind)
                    ind += 1
                    
                    individual = self.grid[i][j]
                    self.model.set_model_params(individual.genome_weights, individual.genome_masks)
                    
                    for t in range(self.n_trials): 
                        self.env.reset()
                        done = False
                        output = self.model.forward(torch.zeros(self.params['n_inputs']))
                        fitness = 0
                        for _ in range(self.params['world_iterations']):
                            # if RENDER:
                            #     env.render('human')
                            if self.params["environment"] == "LunarLander":
                                output = output.argmax()
                            
                            obs, reward, terminated, truncated, _ = self.env.step(output)
                            if terminated or truncated:
                                done=True    
                            # obs, reward, done, _ = self.env.step(output)
                            output = self.model.forward(obs)
                            fitness += reward
                            
                            if done:
                                #print('DEAD')
                                break
                        self.robustness[t][i][j] = fitness
                    
                    # print(self.robustness[i][j])
                    
        self.env.close()
        
        file_save = self.path + self.f_name + '_robustness.npz'
        np.savez(file_save, self.robustness, allow_pickle=True)
        

# file_name = "/run_3/1639722578"

# data = np.load('./fede_results' + file_name + '.npz', allow_pickle=True)

# TRIALS = 500
# WORLD_ITERATIONS = 3000

# if len(data) == 3:
#     grid = data['arr_1']
#     fit = data['arr_0']
#     params = data['arr_2']
# else:
#     ids = data['arr_1']
#     fit = data['arr_0']
#     grid = data['arr_2']
#     params = data['arr_3']

# X = grid.shape[0]
# Y = grid.shape[1]    

# max_fit = []
# for i in range(fit.shape[0]):
#     max_fit.append(np.nanmax(fit[i]))
    


# ancestor = np.zeros((X,Y))
# for i in range(grid.shape[0]):
#     for j in range(grid.shape[1]):
#         if grid[i][j] is not None:
#             ancestor[i][j] = grid[i][j].evolutionary_history[-1][1]
            
# ancestor_trim = np.zeros((X,Y))

# positive_fit = fit[-1]>0
# #print(sum(sum(positive_fit)))
# print(sum(sum(grid != None)))

# for i in range(ancestor_trim.shape[0]):
#     for j in range(ancestor_trim.shape[1]):
#         if positive_fit[i][j]:
#             ancestor_trim[i][j] = ancestor[i][j]
            


# N_INPUTS = 24
# #print(N_INPUTS)
# N_HIDDENS = [40, 40]
# N_OUTPUTS = 4
# IS_MASK = [True, True]
# DISCONNECTABLE_UNITS = 0
# for i in range(len(N_HIDDENS)):
#     if IS_MASK[i]:
#         DISCONNECTABLE_UNITS += N_HIDDENS[i]


# mlp_test = MLP(N_INPUTS, N_HIDDENS, N_OUTPUTS, IS_MASK)
# #mlp_test.set_model_params(individual.genome_weights, individual.genome_masks)

# env = BipedalWalker()
         
# robustness = np.zeros((X,Y))   
# #robustness[:] = np.nan 
# '''test individual'''            
# ind = 0
# for i in range(grid.shape[0]):
#     for j in range(grid.shape[1]):
#         if grid[i][j] is not None:
            
#             print(ind)
#             ind += 1
            
#             individual = grid[i][j]
#             mlp_test.set_model_params(individual.genome_weights, individual.genome_masks)
            
#             for t in range(TRIALS): 
#                 env.reset()
#                 output = mlp_test.forward(torch.zeros(N_INPUTS))
#                 fitness = 0
#                 for _ in range(WORLD_ITERATIONS):
#                     # if RENDER:
#                     #     env.render('human')
#                     obs, reward, done, _ = env.step(output)
#                     output = mlp_test.forward(obs)
#                     fitness += reward
                    
#                     if done:
#                         #print('DEAD')
#                         break
#                 robustness[i][j] += fitness
#             robustness[i][j] /= TRIALS
            
#             print(robustness[i][j])
            
# env.close()

# file_save = './fede_results' + file_name + '_robustness.npz'

# np.savez(file_save, robustness)