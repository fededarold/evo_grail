# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:43:33 2023

@author: darol
"""

from multiprocessing import set_start_method

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

from grail.info_teory_disc import InfoDiscrete
from numpy.matlib import repmat


# class EvoGRAIL_V1():
    
#     def __init__(self):
        


def main_fun_MP(G, current_epoch: int, n_pools: int, epoch_increase=10): #, iterations=10):
    
    N=len(G)
    max_epoch = current_epoch + epoch_increase
    for i in range(N):
        G[i].max_epochs = max_epoch
        
        
    with Pool(n_pools) as pool:     
        
        results = []
        for i in range(N):  # this loop should be ran in multiprocess
            results.append(pool.apply_async(G[i].main, ))

        # Wait for the simulations to complete            
        for result in results:
            result.get()
    
    
    pool.close()
    pool.join()
        
    return G, max_epoch

'''place the guy in the right cell'''
def get_individual_xy_grid(feature_x: float, feature_y: float,
                           feature_x_min=0., feature_x_max=1.,
                           feature_y_min=0., feature_y_max=1.,
                           grid_x=10, grid_y=10):
    
    x_bin = (feature_x_max - feature_x_min) / grid_x
    y_bin = (feature_y_max - feature_y_min) / grid_y
    
    idx = int(np.floor(feature_x / x_bin))        
    idy = int(np.floor(feature_y / y_bin)) 
    if idx == grid_x:
        idx -= 1
    if idy == grid_y:
        idy -= 1
        
    return idx, idy

def init_population(n_agents, randomizer, parallel_execution=True):
    
    genome_list = []
    genome = [0]
    for i in range(n_agents):
        genome_list.append(randomizer.mutate(genome=copy.deepcopy(genome),
                                             param_A=randomizer.initialization_param_A, 
                                             param_B=randomizer.initialization_param_B))
        
    grails_initial_batch = np.empty((n_agents), dtype=object)
    for i in range(n_agents):
        if parallel_execution:
            grails_initial_batch[i] = MOTIVEN.create()
        else:
            grails_initial_batch[i] = MOTIVEN()
        grails_initial_batch[i].set_softmax_temperature(genome_list[i])
        grails_initial_batch[i].agent_id = i
        grails_initial_batch[i].agent_epoch = 0
        grails_initial_batch[i].init_main()
        print("init grail: " + str(i))
    print("done")    
    
    return grails_initial_batch


def place_individuals_in_grid(grails):
    
    grails_map_parents = np.empty((10,10), dtype=object)
    for i in range(len(grails)):
        I_scaled = InfoDiscrete.range_scaler(grails[i].I_sensors_action,
                                             I_lower_bound, I_upper_bound)
        
        x, y = get_individual_xy_grid(feature_x=grails[i].entropy_goal, 
                                      feature_y=I_scaled)
        
        if grails_map_parents[x][y] is None:
            grails_map_parents[x][y] = copy.deepcopy(grails[i])
        else:
            if grails_map_parents[x][y].reward_test < grails[i].reward_test:
               grails_map_parents[x][y] = copy.deepcopy(grails[i])
            elif grails_map_parents[x][y].reward == grails[i].reward:
                if np.random.rand() > 0.5:
                    grails_map_parents[x][y] = copy.deepcopy(grails[i])

    return grails_map_parents


def reproduce(grails_map, randomizer, c_iteration):
    
    grails = []
    # do parents first
    for i in range(grails_map.shape[0]):
        for j in range(grails_map.shape[1]):            
            if grails_map[i][j] is not None:                
                grails.append(copy.deepcopy(grails_map[i][j]))       # maybe we do not need deepcopy
                
    # do offspring
    for i in range(grails_map.shape[0]):
        for j in range(grails_map.shape[1]):            
            if grails_map[i][j] is not None:               
                grails.append(copy.deepcopy(grails_map[i][j]))
                genome = grails[-1].get_softmax_temperature()
                genome = randomizer.mutate(genome=genome,
                                           param_A=randomizer.mutation_param_A, 
                                           param_B=randomizer.mutation_param_B)
                grails[-1].set_softmax_temperature(genome)
                grails[-1].agent_epoch = c_iteration
                
    return grails
    


if __name__ == "__main__":
    
    set_start_method("spawn")
        
    TEST_EVERY_INITIAL_TRAINING = 30
    N_ITERATIONS = 10
    TEST_EVERY = 30
 
    # t_0 = time.time()
    
    random.seed(1)
    np.random.seed(1)

    N_AGENTS = 2
    N_POOLS = 4
    
    uniform = {"initialization": {"lower bound": 0.01,
                                  "upper bound": 0.99},
                "mutation": {"lower bound": -0.5,
                              "upper bound": 0.5},
                "limits": {"min value": 0.01, 
                           "max value": 0.99}}
    
    
    I_upper_history = []
    I_lower_history = []
    
    randomizer = rng.factory_randomizer(distribution="uniform", 
                                        data_type="python", 
                                        params=uniform)

    
    
    # grails_map_parents = np.empty((10,10), dtype=object)
    # grails_map_offspring = np.empty((10,10), dtype=object)
    map_spreading = np.empty((N_ITERATIONS,10,10), dtype=object)
    map_spreading[:] = np.nan
    map_reward = np.zeros((N_ITERATIONS,10,10))
    map_reward[:] = np.nan
    
    grails_initial_batch = init_population(n_agents=N_AGENTS, randomizer=randomizer)
    
    
    grails_initial_batch, current_epoch = main_fun_MP(grails_initial_batch, 
                                                   current_epoch=0,     
                                                   epoch_increase=TEST_EVERY_INITIAL_TRAINING,
                                                   n_pools = N_POOLS) 
    
    #get min-max values of the necessary BD dimensions
    I_values = []
    for i in range(len(grails_initial_batch)):
        I_values.append(grails_initial_batch[i].I_sensors_action)
    I_upper_bound = max(I_values)
    I_lower_bound = min(I_values)
    
    I_upper_history.append(I_upper_bound)
    I_lower_history.append(I_lower_bound)
    
    
    grails_map = place_individuals_in_grid(grails_initial_batch)    
    grails = reproduce(grails_map=grails_map, randomizer=randomizer, c_iteration=1)
    
    for i in range(grails_map.shape[0]):
        for j in range(grails_map.shape[1]):
            if grails_map[i][j] is not None:
                map_spreading[0][i][j] = (grails_map[i][j].agent_id, 
                                          grails_map[i][j].agent_epoch,
                                          grails_map[i][j].goal_manager.softmax_temp[0])
                map_reward[0][i][j] = grails_map[i][j].reward_test
            
    
    for i in range(0,len(grails),N_POOLS):
        print(i)
        grails[i:i+N_POOLS], current_epoch = main_fun_MP(grails[i:i+N_POOLS], 
                                                        current_epoch=current_epoch+1,     
                                                        epoch_increase=TEST_EVERY,
                                                        n_pools = N_POOLS)
        
    for i in range(grails_map.shape[0]):
        for j in range(grails_map.shape[1]):
            if grails_map[i][j] is not None:
                map_spreading[1][i][j] = (grails_map[i][j].agent_id, 
                                          grails_map[i][j].agent_epoch,
                                          grails_map[i][j].goal_manager.softmax_temp[0])
                map_reward[1][i][j] = grails_map[i][j].reward_test
    '''   
    
                
    c_iteration = 0
    f_name = "test_" + str(c_iteration) + ".npz"
    np.savez(f_name,  
              I_upper_history=np.array(I_upper_history),
              I_lower_history=np.array(I_lower_history),
              map_reward=map_reward,
              map_spreading=map_spreading,
              grails=grails_parents,
              allow_pickle=True)
    
    
        
    
    for c_iteration in range(1,N_ITERATIONS):  
        print(c_iteration)
        # do parents
        grails_parents, _ = main_fun(grails_parents, current_epoch=current_epoch+1, epoch_increase=TEST_EVERY) 
        # do offsprings
        grails_offspring, current_epoch = main_fun(grails_offspring, current_epoch=current_epoch+1, epoch_increase=TEST_EVERY)
        
        # reset the grid
        grails_map_parents = np.empty((10,10), dtype=object)
    
        #get min-max values of the necessary BD dimensions for parents and offspring
        I_values = []
        for i in range(len(grails_parents)):
            I_values.append(grails_parents[i].I_sensors_action)
        # keep them separate to see if the BD differs
        for i in range(len(grails_offspring)):
            I_values.append(grails_offspring[i].I_sensors_action)
        I_upper_bound = max(I_values)
        I_lower_bound = min(I_values)
        
        I_upper_history.append(I_upper_bound)
        I_lower_history.append(I_lower_bound)
        
        # assess parents
        for i in range(len(grails_parents)):
            I_scaled = InfoDiscrete.range_scaler(grails_parents[i].I_sensors_action,
                                                 I_lower_bound, I_upper_bound)
            
            x, y = get_individual_xy_grid(feature_x=grails_parents[i].entropy_goal, 
                                          feature_y=I_scaled)
            
            if grails_map_parents[x][y] is None:
                grails_map_parents[x][y] = grails_parents[i]
            else:
                if grails_map_parents[x][y].reward_test < grails_parents[i].reward_test:
                   grails_map_parents[x][y] = grails_parents[i]
                elif grails_map_parents[x][y].reward == grails_parents[i].reward:
                    if np.random.rand() > 0.5:
                        grails_map_parents[x][y] = grails_parents[i]
                        
        # assess offspring
        for i in range(len(grails_offspring)):
            I_scaled = InfoDiscrete.range_scaler(grails_offspring[i].I_sensors_action,
                                                 I_lower_bound, I_upper_bound)
            
            x, y = get_individual_xy_grid(feature_x=grails_offspring[i].entropy_goal, 
                                          feature_y=I_scaled)
            
            if grails_map_parents[x][y] is None:
                grails_map_parents[x][y] = grails_offspring[i]
            else:
                if grails_map_parents[x][y].reward_test < grails_offspring[i].reward_test:
                   grails_map_parents[x][y] = grails_offspring[i]
                elif grails_map_parents[x][y].reward == grails_offspring[i].reward:
                    if np.random.rand() > 0.5:
                        grails_map_parents[x][y] = grails_parents[i]
        
        # reproduce and save data
        grails_offspring = []
        grails_parents = []   
        individual = 0
        for i in range(grails_map_parents.shape[0]):
            for j in range(grails_map_parents.shape[1]):
                if grails_map_parents[i][j] is not None:
                    map_spreading[c_iteration][i][j] = grails_map_parents[i][j].agent_id
                    map_reward[c_iteration][i][j] = grails_map_parents[i][j].reward_test
                    grails_parents.append(copy.deepcopy(grails_map_parents[i][j]))       # maybe we do not need deepcopy
                    grails_offspring.append(copy.deepcopy(grails_map_parents[i][j]))
                    genome = grails_offspring[individual].get_softmax_temperature()
                    genome = randomizer.mutate(genome=copy.deepcopy(genome),
                                               param_A=randomizer.mutation_param_A, 
                                               param_B=randomizer.mutation_param_B)
                    grails_offspring[individual].set_softmax_temperature(genome)
                    grails_offspring[individual].agent_epoch = c_iteration+1
                    individual += 1
        
        f_name = "test_" + str(c_iteration) + ".npz"
        np.savez(f_name, 
                  I_upper_history=np.array(I_upper_history),
                  I_lower_history=np.array(I_lower_history),
                  map_reward=map_reward,
                  map_spreading=map_spreading,
                  grails=grails_parents,
                  allow_pickle=True)
    '''