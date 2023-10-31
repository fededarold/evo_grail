# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:10:53 2023

@author: darol
"""
import numpy as np
import random
from scipy.special import expit, softmax

from neuroevolution import AgentBase, NeuroEvolution
from evo_grail import MLP
import matplotlib.pyplot as plt

import time

np.set_printoptions(suppress=True)

SEED = 11

random.seed(SEED)
np.random.seed(SEED)


TRIALS = 400 #250* 30,40,60,200
GENERATIONS = 50
KNOWLEDGE_INITIAL = -3. #mean
POP_SIZE = 100
N_PARENTS = 50
COMPETENCE_HORIZON = 10
ALPHA = 0.1
REWARD_THR = 0.8

MEANS=np.array([0, 0.5, 1., 1.5]) 
STD = 1. # this is good to map into competence = sigmoid



# knowledge=np.zeros(4)

# goal_switch=TRIALS/4


# agent = AgentBase(genome_size=1, 
#                   random_distribution="uniform", 
#                   genome_data_type="python")

# uniform = {"initialization": {"lower bound": 0.,
#                               "upper bound": 1.},
#             "mutation": {"lower bound": -0.1,
#                           "upper bound": 0.1},
#             "limits": {"min value": 0., 
#                         "max value": 1.}}

# evolution = NeuroEvolution(population_size=POP_SIZE, 
#                             agent=agent, 
#                             tot_generations=GENERATIONS,
#                             randomizer_parameters=uniform,
#                             agents_id=True)

# knowledge = np.zeros((GENERATIONS, POP_SIZE, 4))
# knowledge[:] = KNOWLEDGE_INITIAL
# reward_history = np.zeros(COMPETENCE_HORIZON)

# competence = np.zeros((4, COMPETENCE_HORIZON))
# selection = np.zeros((GENERATIONS, POP_SIZE, 4))

# goal_list = np.arange(0, 4)

# final_fit = []


# count_all=[]
# for _ in range(30):
#     goal_achieved = np.zeros(10)
#     mean = 2
#     std = 0.25
#     count = 0
#     while True:
#         count += 1
#         val = np.random.normal(mean,std)
#         goal_achieved = np.roll(goal_achieved, -1)
#         if val >= 1.:
#             goal_achieved[-1] = 1
#             mean += 0.1*np.mean(goal_achieved)
#         else:
#             goal_achieved[-1] = 0
            
#         if np.mean(goal_achieved)==1:
#             break
#     count_all.append(count)
    
# count_all=np.array(count_all)
# print(count_all.mean())
# print(count_all.std())            




g = [1, 1, 1, 1]
decay=[1.4, 1.3, 1.2, 1.1]
learn_all = []
for i in range(4):
    print(i)
    # time.sleep(3)
    learn = [g[i]]
    count = 0
    while True:
        g[i] *= decay[i]
        learn.append(learn[count] + g[i])        
        count +=1
        print(learn[-1])
        if learn[-1] > 1000:
            break
    learn_all.append(learn)


# def learn(goal, goal_list, knowledge);:
#     result = 

# for gen in range(GENERATIONS):
#     print(gen)
#     for ind in range(POP_SIZE):
#         extrinsic_thr = evolution.get_genome(ind)[0]
#         extrinsic_thr = -1.
#         fitness = np.zeros(4)
#         current_extrinsic_goal=-1
        
#         for trial in range(TRIALS):            
#             if trial%goal_switch==0:
#                 current_extrinsic_goal += 1
#                 # print(current_extrinsic_goal)
#             # if extrinsic_thr > np.random.uniform(): 
#             if extrinsic_thr*np.mean(reward_history) > np.random.uniform(): 
#                 knowledge[gen][ind][current_extrinsic_goal] += goals[current_extrinsic_goal]
#                 competence[current_extrinsic_goal] = np.roll(competence[current_extrinsic_goal],-1)
#                 competence[current_extrinsic_goal][-1] = expit(knowledge[gen][ind][current_extrinsic_goal]  )
#                 selection[gen][ind][current_extrinsic_goal] += 1
#                 if np.random.normal(knowledge[gen][ind][current_extrinsic_goal],STD) >= 1:
#                     fitness[current_extrinsic_goal] += 1
#                     reward_history = np.roll(reward_history, -1)
#                     reward_history[-1] = 1
#                 else:
#                     reward_history = np.roll(reward_history, -1)
#                     reward_history[-1] = 0
#             else:
#                 mean_differential_competence = np.diff(competence).mean(axis=1)
#                 mean_differential_competence = softmax(mean_differential_competence)
#                 current_intrinsic_goal = np.random.choice(goal_list, p=mean_differential_competence)
#                 # current_intrinsic_goal = np.random.choice(goal_list,
#                 #                                           p=softmax(1-competence[:,-1]))
#                 knowledge[gen][ind][current_intrinsic_goal] += goals[current_intrinsic_goal]
#                 competence[current_intrinsic_goal] = np.roll(competence[current_intrinsic_goal],-1)
#                 competence[current_intrinsic_goal][-1] = expit(knowledge[gen][ind][current_intrinsic_goal])
#                 selection[gen][ind][current_intrinsic_goal] += 1
                
#                 if current_intrinsic_goal==current_extrinsic_goal:
#                     if np.random.normal(knowledge[gen][ind][current_extrinsic_goal],STD) >= 1:
#                         fitness[current_extrinsic_goal] += 1
#                         reward_history = np.roll(reward_history, -1)
#                         reward_history[-1] = 1
#                     else:
#                         reward_history = np.roll(reward_history, -1)
#                         reward_history[-1] = 0
                
#         evolution.set_fitness(agent_id=ind, fitness=np.mean(fitness))
#         if gen == GENERATIONS-1:
#             final_fit.append(fitness)
#     evolution.selection_rank(N_PARENTS)      
       
            
# genomes_final = []
# agents_id = evolution.get_agents_id()
# for i in range(evolution._population_size):
#     genomes_final.append(evolution.get_genome(i))

        
# knowledge_last = knowledge[-1]
                        
            
            
# TODO param multiplies mean reward (within some horizon); reward cumulative; novel GA