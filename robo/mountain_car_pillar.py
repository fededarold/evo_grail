# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 15:40:01 2023

@author: darol
"""

import numpy as np
import torch
# import robosuite as suite
import random
import gym

from neuroevolution import AgentBase, NeuroEvolution
from evo_grail import MLP
# import robosuite.environments.manipulation as scenario
import yaml
import time

# import cma

from torch.nn.utils import parameters_to_vector

SEED = 11

f_name_evo = "neg_rew_pop_sigmoid"
f_name_fit = "neg_rew_fit_sigmoid"

tot_time = time.time()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

HORIZON = 500
RENDER = False
GENERATIONS_TOT = 50
TRIALS_TOT = 10
POPULATION_SIZE = 300
AUTOGRAD = False
SAVE_GEN = 20

N_HIDDEN = 40


# class SampledSoftmax():
    
#     torch.nn.SoftMax(x)
#     dist=torch.distributions.categorical.Categorical(probs=x)
#     return dist.sample().item()


env=gym.make('MountainCar-v0') #,render_mode="human")
mlp = MLP(input_size=env.observation_space.shape[0],
          output_size=env.action_space.n,
          hidden_size=N_HIDDEN,
          activation_function='relu',
          activation_output=torch.nn.Softmax())

agent=AgentBase(genome_size=mlp.get_parameters_number())#, 
                # random_distribution="uniform")

normal = {"initialization": {"mean": 0.,
                             "std": 1.},
            "mutation": {"mean": 0.,
                         "std": 0.01}}

uniform = {"initialization": {"lower bound": 0.,
                              "upper bound": 1.},
            "mutation": {"lower bound": -0.01,
                          "upper bound": 0.01},
            "limits": {"min value": -10., 
                       "max value": 10.}}

evolution = NeuroEvolution(population_size=POPULATION_SIZE, 
                           agent=agent, 
                           agents_id=True,
                           randomizer_parameters=normal)


nn_output = np.zeros(env.action_space.n)
nn_input_zeros = torch.zeros(env.observation_space.shape[0])
action = 1
for gen in range(GENERATIONS_TOT):
    
    time_start = time.time()
    best_fit = -10000
    best_dist = -10000
    print(gen)
    for ind in range(evolution._population_size):
        # print(ind)
        # map genotype into the controller
        mlp.set_parameters(params=evolution.get_genome(agent_id=ind))
        # print(parameters_to_vector(nn.parameters()))
        fitness = 0.
        max_dist = -10
        
        # env.reset()
        fitness_vel = 0.
        fitness_dist = 0.
        for t in range(TRIALS_TOT):
            # reset the environment
            env.reset()        
            # reward_trial = np.zeros(HORIZON)
            nn_output = mlp(nn_input_zeros) # sample random action
            touched = False
            for i in range(HORIZON):    
                
                # print(nn_output)
                
                obs, reward, done, info = env.step(action)  # take action in the environment  
                nn_output = mlp(torch.tensor(obs))
                dist=torch.distributions.categorical.Categorical(probs=nn_output)
                action=dist.sample().item()
                # print(action)
                # fitness += reward * np.abs(0.5-obs[0])
                fitness_vel += np.abs(obs[1]) * 10
                fitness_dist += obs[0]
                
                if max_dist < obs[0]:
                    max_dist=obs[0]
                
                # print(obs)
                
                # print(reward)
                
                if RENDER:
                    env.render()  # render on display
                    
                if done:
                    break
            
            
           
            if best_dist < max_dist: #fitness:
                best_dist=max_dist #fitness
            fitness_vel /= i
            # fitness_dist /= i
            # fitness += fitness_vel + max_dist#fitness_dist
            fitness += max_dist
        
        # fitness /= t
        if best_fit < fitness:
            best_fit=fitness
        evolution.set_fitness(agent_id=ind, fitness=fitness)
        # fitness_debug[gen][ind] = fitness
            # print("rew: " + str(fit) + " done: " + str(done) + " ts: " + str(i))
    # print(f"Fit: max: {fitness_debug[gen].max():.5f} min: {fitness_debug[gen].min():.5f} mean: {fitness_debug[gen].mean():.5f}")
    print("fit " + str(best_fit))
    print("dist " + str(best_dist))
    evolution.selection_rank(n_parents=150)    
    # print("Time:" + str((time.time()-time_start)/60))
    # if gen%SAVE_GEN == 0:
    #     f_n_fit = f_name_fit + "_" + str(gen)
    #     f_n_evo = f_name_evo + "_" + str(gen)
    #     evolution.save_evolution(f_n_evo)
    #     np.savez(file=f_n_fit,
    #              fitness_history=fitness_debug,
    #              allow_pickle=True)
env.close()

#%%
'''
for ind in range(evolution._population_size):
        # print(ind)
        # map genotype into the controller
        mlp.set_parameters(params=evolution.get_genome(agent_id=ind))
        # print(parameters_to_vector(nn.parameters()))
        fitness = 0.
        max_dist = -10
        
        # env.reset()
        
        for t in range(TRIALS_TOT):
            # reset the environment
            env.reset()        
            # reward_trial = np.zeros(HORIZON)
            nn_output = mlp(nn_input_zeros) # sample random action
            touched = False
            for i in range(HORIZON):    
                
                # print(nn_output)
                
                obs, reward, done, info = env.step(action)  # take action in the environment  
                nn_output = mlp(torch.tensor(obs))
                dist=torch.distributions.categorical.Categorical(probs=nn_output)
                action=dist.sample().item()
                # print(action)
                # fitness += reward * np.abs(0.5-obs[0])
                if max_dist < obs[0]:
                    max_dist=obs[0]
                
                # print(obs)
                
                # print(reward)
                
                # if RENDER:
                env.render()  # render on display
                    
                if done:
                    break
            
            if best_fit < max_dist: #fitness:
                best_fit=max_dist #fitness
            # print (ind)
            # print (i)
            # print (fitness)
        
        # evolution.set_fitness(agent_id=ind, fitness=fitness)
        # fitness_debug[gen][ind] = fitness
            # print("rew: " + str(fit) + " done: " + str(done) + " ts: " + str(i))
    # print(f"Fit: max: {fitness_debug[gen].max():.5f} min: {fitness_debug[gen].min():.5f} mean: {fitness_debug[gen].mean():.5f}")
    # print(best_fit)
    # evolution.selection_rank(n_parents=30)    
    # print("Time:" + str((time.time()-time_start)/60))
    # if gen%SAVE_GEN == 0:
    #     f_n_fit = f_name_fit + "_" + str(gen)
    #     f_n_evo = f_name_evo + "_" + str(gen)
    #     evolution.save_evolution(f_n_evo)
    #     np.savez(file=f_n_fit,
    #              fitness_history=fitness_debug,
    #              allow_pickle=True)
env.close()

'''


