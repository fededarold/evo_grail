
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 18:15:25 2023

@author: darol
"""


from multiprocessing import Pool
from multiprocessing.managers import NamespaceProxy, BaseManager
# from multiprocess import Pool
# from multiprocess.managers import NamespaceProxy, BaseManager
import types
import numpy as np
import time
import robo.neuroevolution as ga
import robo.randomizer as rng
import random

from grail.motiven_bandit_ale_MP import MOTIVEN

import copy

from grail.info_teory_disc import information_theory_discrete
from numpy.matlib import repmat


def main_fun(G, current_epoch: int, epoch_increase=30): #, iterations=10):
    
    N=len(G)
    max_epoch = current_epoch + epoch_increase
    for i in range(N):
        G[i].max_epochs = max_epoch
        G[i].main()
        
    return G, max_epoch
    

# if __name__ == "__main__":

t_0 = time.time()
    
random.seed(1)
np.random.seed(1)

n_agents = 5
uniform = {"initialization": {"lower bound": 0.01,
                              "upper bound": 0.99},
            "mutation": {"lower bound": -0.01,
                          "upper bound": 0.01},
            "limits": {"min value": 0.01, 
                       "max value": 0.99}}

randomizer = rng.factory_randomizer(distribution="uniform", 
                                    data_type="python", 
                                    params=uniform)

genome = [0]
genome_list = []
for i in range(n_agents):
    genome_list.append(randomizer.mutate(genome=copy.deepcopy(genome),
                                         param_A=randomizer.initialization_param_A, 
                                         param_B=randomizer.initialization_param_B))

grails = []
for i in range(n_agents):
    grails.append(MOTIVEN())
    grails[i].set_softmax_temperature(genome_list[i])
    grails[i].init_main()

current_epoch = -1 
for _ in range(3):  
    print(current_epoch)
    grails, current_epoch = main_fun(grails, current_epoch=current_epoch+1, epoch_increase=10) 
    for i in range(n_agents):
        grails[i].get_data()


entropy_goal = np.zeros(len(grails))
entropy_sensors = np.zeros(len(grails))
entropy_action = np.zeros(len(grails))
I_sensors_action = np.zeros(len(grails))

for i in range(len(grails)):
    entropy_goal[i] = information_theory_discrete.entropy(data=grails[i].goal_data, 
                                                        n_bins=6, 
                                                        states_range=[1,6],
                                                        normalize=False)
    
    entropy_sensors[i] = information_theory_discrete.entropy(data=grails[i].sensors_data.flatten(), 
                                                          n_bins=30,
                                                          states_range=[0.,1],
                                                          normalize=False)
    
    entropy_action[i] = information_theory_discrete.entropy(data=grails[i].action_data, 
                                                          n_bins=30,
                                                          states_range=[0.,1],
                                                          normalize=False)
    
    action_expanded = repmat(grails[i].action_data, n=1, m=6).T
    entropy_sensors_action = information_theory_discrete.joint_entropy(data_x=action_expanded.flatten(), 
                                                                       data_y=grails[i].sensors_data.flatten(), 
                                                                       n_bins=[30, 30], 
                                                                       states_range=[[0.,1.],[0.,1.]])
    # print(entropy_sensors_action)
    I_sensors_action[i] = information_theory_discrete.mutual_information(E_x=entropy_action[i],
                                                                      E_y=entropy_sensors[i],
                                                                      E_xy=entropy_sensors_action,
                                                                      normalization="IQR")
    
entropy_goal /= np.log(6)
entropy_action /= np.log(30)
entropy_sensors /= np.log(30)

print(time.time()-t_0)

# TODO: use parameters in main: get the current epoch from G[0] and pass it
