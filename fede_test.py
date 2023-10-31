# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:27:55 2023

@author: darol
"""
import numpy as np
import torch
import robosuite as suite
import random

from neuroevolution import AgentBase, NeuroEvolution
from evo_grail import MLP
# import robosuite.environments.manipulation as scenario
import yaml
import time

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
GENERATIONS_TOT = 500
TRIALS_TOT = 5
POPULATION_SIZE = 100
AUTOGRAD = False
SAVE_GEN = 20

# robot_initial_pose = [ 0.  ,  0.65,  0.  , -1.89,  0.  ,  0.6 ,  0.  ]
robot_initial_pose = [ 0.  ,  0.,  0.  , -1.89,  0.  ,  0.6 ,  0.  ]

fitness_debug = np.zeros((GENERATIONS_TOT, POPULATION_SIZE))

# torch.autograd.set_grad_enabled(False)

# create environment instance
env = suite.make(
    env_name="Pillar2", # try with other tasks like "Stack" and "Door"
    # env_name="PickPlace",
    robots="IIWA",  # try with other robots like "Sawyer" and "Jaco"
    gripper_types="default",
    initial_joints_position=robot_initial_pose,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    reward_shaping=True
)

input_size = 6
output_size = len(env.action_spec[0])-1
# create controller: action space excluding gripper as we keep it closed for now
nn = MLP(input_size=input_size, 
         output_size=output_size,
         activation_function="sigmoid")
if not AUTOGRAD:
    nn.requires_grad_(False)
# define agent
agent = AgentBase(genome_size=nn.get_parameters_number())
# create population
normal = {"initialization": {"mean": 0.,
                             "std": 0.1},
            "mutation": {"mean": 0.,
                         "std": 0.001}}
evolution = NeuroEvolution(population_size=POPULATION_SIZE, 
                           agent=agent, 
                           agents_id=True,
                           tot_generations=GENERATIONS_TOT,
                           randomizer_parameters=normal)

nn_output = np.zeros(env.robots[0].dof)
nn_output[-1] = 1.
nn_input_zeros = torch.zeros(input_size)
for gen in range(GENERATIONS_TOT):
    
    time_start = time.time()
    
    for ind in range(evolution._population_size):
        # print(ind)
        # map genotype into the controller
        nn.set_parameters(params=evolution.get_genome(agent_id=ind))
        # print(parameters_to_vector(nn.parameters()))
        fitness = 0.
        for t in range(TRIALS_TOT):
            # reset the environment
            env.reset()        
            reward_trial = np.zeros(HORIZON)
            nn_output[:-1] = nn(nn_input_zeros) # sample random action
            touched = False
            for i in range(HORIZON):    
                
                # print(nn_output)
                
                obs, reward, done, info = env.step(nn_output)  # take action in the environment  
                positions = np.concatenate((obs["objectBottle_pos"], obs["robot0_eef_pos"]))
                positions = torch.Tensor(positions)
                nn_output[:-1] = nn(positions)
                reward_trial[i] = reward
                if RENDER:
                    env.render()  # render on display
                if done and not touched:
                    touched = True
                    # we get the past average shaped reward and we sum the final succesfull state
                    # reward_trial = np.nan_to_num(reward_trial)
                    # if i != 0:
                    fitness += 1. # reward_trial[i] - np.mean(reward_trial[:i]) 
                    # else:
                    #     print("problem")
                    # break
            # if not done:
                # we just compute the average distance from the object
                # reward_trial = np.nan_to_num(reward_trial)
            fitness -= np.mean(reward_trial)
            # print(i)
        evolution.set_fitness(agent_id=ind, fitness=fitness)
        fitness_debug[gen][ind] = fitness
            # print("rew: " + str(fit) + " done: " + str(done) + " ts: " + str(i))
    print(f"Fit: max: {fitness_debug[gen].max():.5f} min: {fitness_debug[gen].min():.5f} mean: {fitness_debug[gen].mean():.5f}")
    evolution.selection_rank(n_parents=50)    
    print("Time:" + str((time.time()-time_start)/60))
    if gen%SAVE_GEN == 0:
        f_n_fit = f_name_fit + "_" + str(gen)
        f_n_evo = f_name_evo + "_" + str(gen)
        evolution.save_evolution(f_n_evo)
        np.savez(file=f_n_fit,
                 fitness_history=fitness_debug,
                 allow_pickle=True)
env.close()

# TODO save model
# TODO save environment
evolution.save_evolution(f_name_evo)
np.savez(file=f_name_fit,
         fitness_history=fitness_debug,
         allow_pickle=True)

final_time = time.time()
print("tot time: " + str((time.time() - tot_time)/3600))
         

# pop_history = evolution._population_history
# ids = np.zeros((pop_history.shape[0],pop_history.shape[1]))
# for i in range(pop_history.shape[0]):
#     for j in range(pop_history.shape[1]):
#         ids[i][j] = pop_history[i][j]._agent_id

# for i in range(pop_history.shape[0]):
#     print(np.unique(ids[i]).size)
    
    
# fit_final = fitness_debug[-1]
# bests = np.where(fit_final > TRIALS_TOT-1)
# for b_ind in range(len((bests[0]))):
#     # nn.set_parameters(params=evolution.get_genome(agent_id=b_ind))
#     nn.set_parameters(params=pop_history[0][15].genome)
#     fitness = 0.
#     for t in range(5):
#         # reset the environment
#         env.reset()     
        
#         nn_output[:-1] = nn(nn_input_zeros) # sample random action
#         reward_trial = np.zeros(HORIZON)
        
#         for i in range(HORIZON):            
#             obs, reward, done, info = env.step(nn_output)  # take action in the environment  
#             positions = np.concatenate((obs["objectBottle_pos"], obs["robot0_eef_pos"]))
#             positions = torch.Tensor(positions)
#             nn_output[:-1] = nn(positions)
#             env.render()  # render on display
#             reward_trial[i] = reward
#             # if RENDER:
#             env.render()  # render on display
            
#             print(reward)
            
#             if done:
#                 # we get the past average shaped reward and we sum the final succesfull state
#                 # reward_trial = np.nan_to_num(reward_trial)
#                 if i != 0:
#                     fitness += np.mean(reward_trial[:i]) + reward_trial[i]
#                 else:
#                     print("problem")
#                 break
#         if not done:
#             # we just compute the average distance from the object
#             # reward_trial = np.nan_to_num(reward_trial)
#             fitness += np.mean(reward_trial)
#         print(fitness)
# env.close()
