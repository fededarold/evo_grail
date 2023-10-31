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


SEED = 11

random.seed(SEED)
np.random.seed(SEED)


TRIALS = 500
GENERATIONS = 50
KNOWLEDGE_INITIAL = -3. #we use sigmoid
POP_SIZE = 100
N_PARENTS = 50
COMPETENCE_HORIZON = 10
ALPHA = 0.1
REWARD_THR = 0.8

goals=np.array([[10., 1.5], # 45 trials
                [10., 5.],  # 55 trials
                [5., 10.],  # 120 trials
                [1.5, 10.]]) # 300 trials

goal_switch=TRIALS/4

agent = AgentBase(genome_size=1, 
                  random_distribution="uniform", 
                  genome_data_type="python")

uniform = {"initialization": {"lower bound": 0.,
                              "upper bound": 1.},
            "mutation": {"lower bound": -0.1,
                          "upper bound": 0.1},
            "limits": {"min value": 0., 
                       "max value": 1.}}

evolution = NeuroEvolution(population_size=POP_SIZE, 
                           agent=agent, 
                           tot_generations=GENERATIONS,
                           randomizer_parameters=uniform,
                           agents_id=True)

knowledge = np.zeros((GENERATIONS, POP_SIZE, 4))
knowledge[:] = KNOWLEDGE_INITIAL

competence = np.zeros((4, COMPETENCE_HORIZON))
selection = np.zeros((GENERATIONS, POP_SIZE, 4))

goal_list = np.arange(0, 4)

final_fit = []

for gen in range(GENERATIONS):
    print(gen)
    for ind in range(POP_SIZE):
        extrinsic_thr = evolution.get_genome(ind)[0]
        fitness = np.zeros(4)
        current_extrinsic_goal=-1
        
        for trial in range(TRIALS):            
            if trial%goal_switch==0:
                current_extrinsic_goal += 1
                # print(current_extrinsic_goal)
            if extrinsic_thr > np.random.uniform():
                knowledge_new = ALPHA * np.random.beta(goals[current_extrinsic_goal][0], 
                                                       goals[current_extrinsic_goal][1])
                knowledge[gen][ind][current_extrinsic_goal] += knowledge_new
                competence[current_extrinsic_goal] = np.roll(competence[current_extrinsic_goal],-1)
                competence[current_extrinsic_goal][-1] = expit(knowledge[gen][ind][current_extrinsic_goal]  )
                selection[gen][ind][current_extrinsic_goal] += 1
                if competence[current_extrinsic_goal][-1] > REWARD_THR:
                    if current_extrinsic_goal == 3:
                        fitness[current_extrinsic_goal] += 10
                else:
                    fitness[current_extrinsic_goal] += 1
                # else:
                #     fitness[current_extrinsic_goal] = competence[current_extrinsic_goal][-1] 
            else:
                mean_differential_competence = np.diff(competence).mean(axis=1)
                mean_differential_competence = softmax(mean_differential_competence)
                current_intrinsic_goal = np.random.choice(goal_list, p=mean_differential_competence)
                knowledge_new = ALPHA * np.random.beta(goals[current_intrinsic_goal][0], 
                                                       goals[current_intrinsic_goal][1])
                knowledge[gen][ind][current_intrinsic_goal] += knowledge_new
                competence[current_intrinsic_goal] = np.roll(competence[current_intrinsic_goal],-1)
                competence[current_intrinsic_goal][-1] = expit(knowledge[gen][ind][current_intrinsic_goal])
                selection[gen][ind][current_intrinsic_goal] += 1
        evolution.set_fitness(agent_id=ind, fitness=np.mean(fitness))
        if gen == GENERATIONS-1:
            final_fit.append(fitness)
    evolution.selection_rank(N_PARENTS)      
       
            
genomes_final = []
agents_id = evolution.get_agents_id()
for i in range(evolution._population_size):
    genomes_final.append(evolution.get_genome(i))

        
knowledge_last = knowledge[-1]
                        
            
            
# TODO param multiplies mean reward (within some horizon); reward cumulative; novel GA