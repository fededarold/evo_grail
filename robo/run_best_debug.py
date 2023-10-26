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


HORIZON=300

f_name_evo = "neg_rew_pop_relu"
f_name_fit = "neg_rew_fit_relu"

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

data_evo = np.load("neg_rew_pop_sig.npz", allow_pickle=True)
data_fit = np.load("neg_rew_fit_sig.npz", allow_pickle=True)
pop_history = data_evo["population"]
fit_history = data_fit["fitness_history"]


# pop_history = evolution._population_history
ids = np.zeros((pop_history.shape[0],pop_history.shape[1]))
for i in range(pop_history.shape[0]):
    for j in range(pop_history.shape[1]):
        ids[i][j] = pop_history[i][j]._agent_id

for i in range(pop_history.shape[0]):
    print(np.unique(ids[i]).size)


nn_output = np.zeros(env.robots[0].dof)
nn_output[-1] = 0.
nn_input_zeros = torch.zeros(6)
    
    
fit_final = fit_history[-1]
# bests = np.where(fit_final > 1.)
es_param_vect = len(parameters_to_vector(nn.parameters()))
es = cma.CMAEvolutionStrategy(x0=len(parameters_to_vector(nn.parameters()))*[0],
                              sigma0=0.1,
                              inopts={'popsize': 40})

for b_ind in range(len(fit_final)):
    # nn.set_parameters(params=evolution.get_genome(agent_id=b_ind))
    nn.set_parameters(params=pop_history[0][15].genome)
    fitness = 0.
    for t in range(5):
        # reset the environment
        env.reset()     
        
        nn_output[:-1] = nn(nn_input_zeros) # sample random action
        reward_trial = np.zeros(HORIZON)
        
        for i in range(HORIZON):            
            if i == HORIZON-1:
                print("debug")
            
            obs, reward, done, info = env.step(nn_output)  # take action in the environment  
            positions = np.concatenate((obs["objectBottle_pos"], obs["robot0_eef_pos"]))
            positions = torch.Tensor(positions)
            nn_output[:-1] = nn(positions)
            env.render()  # render on display
            reward_trial[i] = reward
            # if RENDER:
            env.render()  # render on display
            
            # print(reward)
            
            if done:
                # we get the past average shaped reward and we sum the final succesfull state
                # reward_trial = np.nan_to_num(reward_trial)
                if i != 0:
                    fitness += np.mean(reward_trial[:i]) + reward_trial[i]
                else:
                    print("problem")
                break
        if not done:
            # we just compute the average distance from the object
            # reward_trial = np.nan_to_num(reward_trial)
            fitness += np.mean(reward_trial)
        print(fitness)
env.close()
