# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:05:50 2023

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

import cma

from torch.nn.utils import parameters_to_vector, vector_to_parameters


HORIZON=500

f_name_evo = "CMAES_60"


robot_initial_pose = [ 0.  ,  0.,  0.  , -1.89,  0.  ,  0.6 ,  0.  ]

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

output_size = len(env.action_spec[0])-1
nn = MLP(input_size=6, 
         output_size=output_size)
nn.requires_grad_(False)
# np.savez(file=f_name_fit,
#          fitness_history=fitness_debug,
#          allow_pickle=True)

# np.savez(file=file_name,
#          fitness=self._fitness,
#          population=self._population_history,
#          allow_pickle=True)

data = np.load(f_name_evo + ".npz", allow_pickle=True)
# data_fit = np.load("neg_rew_fit_sig.npz", allow_pickle=True)
pop_history = data["individuals"]
fit_history = data["fitness_history"]


# pop_history = evolution._population_history
# ids = np.zeros((pop_history.shape[0],pop_history.shape[1]))
# for i in range(pop_history.shape[0]):
#     for j in range(pop_history.shape[1]):
#         ids[i][j] = pop_history[i][j]._agent_id

# for i in range(pop_history.shape[0]):
#     print(np.unique(ids[i]).size)


nn_output = np.zeros(env.robots[0].dof)
nn_output[-1] = 0.
nn_input_zeros = torch.zeros(6)
    
    
fit_final = fit_history[60]
# bests = np.where(fit_final > 1.)
# es_param_vect = len(parameters_to_vector(nn.parameters()))
# es = cma.CMAEvolutionStrategy(x0=len(parameters_to_vector(nn.parameters()))*[0],
#                               sigma0=0.1,
#                               inopts={'popsize': 40})


fitness_gen = np.zeros(100)
for ind in range(len(pop_history)):
    # nn.set_parameters(params=evolution.get_genome(agent_id=b_ind))
    print(ind)
    ind = 42
    nn.set_parameters(params=torch.Tensor(pop_history[ind]))
    # fitness = 0.
    
    for t in range(5):
        # reset the environment
        print(ind)
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
        
            env.render()  # render on display
            if done and not touched:
                touched = True
                print("touched")
                # we get the past average shaped reward and we sum the final succesfull state
                # reward_trial = np.nan_to_num(reward_trial)
                # if i != 0:
                fitness_gen[ind] += 1. # reward_trial[i] - np.mean(reward_trial[:i]) 
                # else:
                #     print("problem")
                # break
        # if not done:
            # we just compute the average distance from the object
            # reward_trial = np.nan_to_num(reward_trial)
        fitness_gen[ind] -= np.mean(reward_trial)
        
    print(fitness_gen[ind])
    # fitness_debug[gen][ind] = fitness_gen[ind]
env.close()

def logistic(k):
    x = np.arange(-10, 11, 1, dtype=np.float64)
    x *= k
