# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:49:45 2023

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
import neuroevolution as ga
import randomizer as rng
import random

from motiven_bandit_ale_MP import MOTIVEN

import copy

from info_teory_disc import InfoDiscrete
from numpy.matlib import repmat


# class EvoGRAIL_V1():
    
#     def __init__(self):
        


def main_fun_MP(G, current_epoch: int, n_pools: int, epoch_increase=10): #, iterations=10):
    
    N=len(G)
    max_epoch = current_epoch + epoch_increase
    # for i in range(N):
    #     G[i].set_max_epochs(max_epoch)
        
        
    with Pool(n_pools) as pool:     
        
        results = []
        for i in range(N):  # this loop should be ran in multiprocess
            results.append(pool.apply_async(G[i].main, (max_epoch,)))

        # Wait for the simulations to complete            
        for result in results:
            result.get()
    
    
    pool.close()
    pool.join()
        
    return G, max_epoch

def main_fun(G, current_epoch: int, epoch_increase=10):
    
    N=len(G)
    max_epoch = current_epoch + epoch_increase
    for i in range(N):
        G[i].max_epochs = max_epoch
        G[i].main()
        
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

def init_population(n_agents, randomizer_temp, randomizer_beta, parallel_execution=True):
    
    genome_list_temp = []
    genome_list_beta = []
    genome_temp = [0]
    genome_beta = [0]
    for i in range(n_agents):
        if randomizer_temp is not None:
            genome_list_temp.append(randomizer_temp.mutate(genome=copy.deepcopy(genome_temp),
                                                         param_A=randomizer_temp.initialization_param_A, 
                                                         param_B=randomizer_temp.initialization_param_B))
        if randomizer_beta is not None:
            genome_list_beta.append(randomizer_beta.mutate(genome=copy.deepcopy(genome_beta),
                                                         param_A=randomizer_beta.initialization_param_A, 
                                                         param_B=randomizer_beta.initialization_param_B))
    
    grails_initial_batch = np.empty((n_agents), dtype=object)
    for i in range(n_agents):
        
        if parallel_execution:
            grails_initial_batch[i] = MOTIVEN.create()
        else:
            grails_initial_batch[i] = MOTIVEN()
            
        if randomizer_temp is not None:
            grails_initial_batch[i].set_softmax_temperature(genome_list_temp[i][0])
        else:
            grails_initial_batch[i].set_softmax_temperature(0.02)
        if randomizer_beta is not None:    
            grails_initial_batch[i].set_beta_val(genome_list_beta[i][0])
        else:
            grails_initial_batch[i].set_beta_val(0.3)
        grails_initial_batch[i].agent_id = i
        grails_initial_batch[i].set_agent_epoch(0)
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
            elif grails_map_parents[x][y].reward_test == grails[i].reward_test:
                if np.random.rand() > 0.5:
                    grails_map_parents[x][y] = copy.deepcopy(grails[i])

    return grails_map_parents


def reproduce(grails_map, randomizer_temp, randomizer_beta, c_iteration):
    
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
                
                if randomizer_temp is not None:
                    genome_temp = grails[-1].get_softmax_temperature()
                    genome_temp = randomizer_temp.mutate(genome=[genome_temp],
                                                       param_A=randomizer_temp.mutation_param_A, 
                                                       param_B=randomizer_temp.mutation_param_B)
                    grails[-1].set_softmax_temperature(genome_temp[0])
                
                if randomizer_beta is not None:
                    genome_beta = grails[-1].get_beta_val()
                    genome_beta = randomizer_beta.mutate(genome=[genome_beta],
                                                       param_A=randomizer_beta.mutation_param_A, 
                                                       param_B=randomizer_beta.mutation_param_B)
                    grails[-1].set_beta_val(genome_beta[0])
                    
                grails[-1].set_agent_epoch(c_iteration)
                
    return grails


def fetch_data(grails_map):
    
    goals_list = ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'ball_6']
    tot_trials = 0
    
    map_spreading = np.empty((10,10), dtype=object)
    map_spreading[:] = np.nan
    map_reward = np.zeros((10,10))
    map_reward[:] = np.nan
    map_competence = np.zeros((10,10), dtype=object)
    map_competence[:] = np.nan
    for i in range(grails_map.shape[0]):
        for j in range(grails_map.shape[1]):
            if grails_map[i][j] is not None:
                map_spreading[i][j] = (grails_map[i][j].agent_id, 
                                          grails_map[i][j].agent_epoch,
                                          grails_map[i][j].goal_manager.softmax_temp,
                                          grails_map[i][j].goal_manager.beta_val)
                map_reward[i][j] = grails_map[i][j].reward_test
                tot_trials += grails_map[i][j].tot_trials
                competence = []
                for g in range(len(goals_list)):
                    competence.append(grails_map[i][j].goal_manager.goals_competence[goals_list[g]]["c0"])
                map_competence[i][j] = competence
                
    return map_spreading, map_reward, map_competence, tot_trials               
    

def check_competence(competence):
    
    for i in range(competence.shape[0]):
        for j in range(competence.shape[1]):
            if type(competence[i][j]) == list:
                if sum(competence[i][j]) >= 1:
                    return True
    
    return False


if __name__ == "__main__":
    
    set_start_method("spawn")
        
    TEST_EVERY_INITIAL_TRAINING = 50
    N_ITERATIONS = 100
    TEST_EVERY = 50
    
    MP = False
    
    N_AGENTS = 10
    N_POOLS = 20
    
    # uniform_temp = None
    uniform_temp = {"initialization": {"lower bound": 0.01,
                                 "upper bound": 0.99},
                   "mutation": {"lower bound": -0.1,
                                 "upper bound": 0.1},
                   "limits": {"min value": 0.01, 
                               "max value": 0.99}}
    
    uniform_beta = None
    # uniform_beta = {"initialization": {"lower bound": 0.1,
    #                                "upper bound": 0.5},
    #                  "mutation": {"lower bound": -0.1,
    #                                "upper bound": 0.1},
    #                  "limits": {"min value": 0.1, 
    #                             "max value": 0.5}}
    
    randomizer_temp = None
    if uniform_temp is not None:
        randomizer_temp = rng.factory_randomizer(distribution="uniform", 
                                                data_type="python", 
                                                params=uniform_temp)
    
    randomizer_beta = None
    if uniform_beta is not None:
        randomizer_beta = rng.factory_randomizer(distribution="uniform", 
                                                data_type="python", 
                                                params=uniform_beta)
 
    # t_0 = time.time()
    
    random.seed(1)
    np.random.seed(1)

    
    
    hyperparameter = {"init_agents": N_AGENTS,
                      "initial_training": TEST_EVERY_INITIAL_TRAINING,
                      "test_every": TEST_EVERY,
                      "iterations": N_ITERATIONS,
                      "randomizer_temperature": uniform_temp,
                      "randomizer_beta": uniform_beta}
    
    
    
    tot_trials = 0
    I_upper_history = []
    I_lower_history = []
    map_spreading = np.empty((N_ITERATIONS,10,10), dtype=object)
    map_spreading[:] = np.nan
    map_reward = np.zeros((N_ITERATIONS,10,10))
    map_reward[:] = np.nan
    map_competence = np.zeros((N_ITERATIONS,10,10), dtype=object)
    map_competence[:] = np.nan
    
    
    # initial batch
    grails = init_population(n_agents=N_AGENTS, 
                             randomizer_temp=randomizer_temp, 
                             randomizer_beta=randomizer_beta,
                             parallel_execution=MP)
    
    # for i in range(len(grails)):
    #     print(grails[i].goal_manager.beta_val)
    
    if MP:
        for i in range(0,len(grails),N_POOLS):
            grails[i:i+N_POOLS], current_epoch = main_fun_MP(grails[i:i+N_POOLS], 
                                                            current_epoch=0,     
                                                            epoch_increase=TEST_EVERY_INITIAL_TRAINING,
                                                            n_pools = N_POOLS) 
    else:
        grails, current_epoch = main_fun(grails, 
                                         current_epoch=0,     
                                         epoch_increase=TEST_EVERY_INITIAL_TRAINING) 
    
    #get min-max values of the necessary BD dimensions
    I_values = []
    for i in range(len(grails)):
        I_values.append(grails[i].I_sensors_action)
    I_upper_bound = max(I_values)
    I_lower_bound = min(I_values)
    
    I_upper_history.append(I_upper_bound)
    I_lower_history.append(I_lower_bound)
    
    
    grails_map = place_individuals_in_grid(grails) 
    map_spreading[0], map_reward[0], map_competence[0], tot_trials = fetch_data(grails_map)
    grails = reproduce(grails_map=grails_map, 
                       randomizer_temp=randomizer_temp, 
                       randomizer_beta=randomizer_beta,
                       c_iteration=1)    
    
    c_iteration = 0
    f_name = "./results/test_" + str(c_iteration) + ".npz"
    np.savez(f_name,  
              I_upper_history=np.array(I_upper_history),
              I_lower_history=np.array(I_lower_history),
              map_reward=map_reward,
              map_spreading=map_spreading,
              map_competence=map_competence,
              # grails=grails_map,
              allow_pickle=True)
    
    # print(current_epoch)
    for c_iteration in range(1,N_ITERATIONS):  
        print("iteration: " + str(c_iteration))      
        print("agents: " + str(len(grails)))
        if MP:
            for i in range(0,len(grails),N_POOLS):
                # print(i)
                grails[i:i+N_POOLS], current_epoch = main_fun_MP(grails[i:i+N_POOLS], 
                                                                current_epoch=current_epoch+1,     
                                                                epoch_increase=TEST_EVERY,
                                                                n_pools = N_POOLS)
        else:
            grails, current_epoch = main_fun(grails, 
                                            current_epoch=current_epoch+1,     
                                            epoch_increase=TEST_EVERY)
        
        # print(current_epoch)
        
        #get min-max values of the necessary BD dimensions
        I_values = []
        for i in range(len(grails)):
            I_values.append(grails[i].I_sensors_action)
        I_upper_bound = max(I_values)
        I_lower_bound = min(I_values)
        
        I_upper_history.append(I_upper_bound)
        I_lower_history.append(I_lower_bound)
        
        
        grails_map = place_individuals_in_grid(grails)    
        map_spreading[c_iteration], map_reward[c_iteration], map_competence[c_iteration], trials = fetch_data(grails_map)
        tot_trials += trials
        
        if check_competence(map_competence[-1]):
            break
        
        grails = reproduce(grails_map=grails_map, 
                           randomizer_temp=randomizer_temp,
                           randomizer_beta=randomizer_beta,
                           c_iteration=c_iteration+1)
        
        
        f_name = "./results/test_" + str(c_iteration) + ".npz"
        np.savez(f_name,  
                  I_upper_history=np.array(I_upper_history),
                  I_lower_history=np.array(I_lower_history),
                  map_reward=map_reward,
                  map_spreading=map_spreading,
                  map_competence=map_competence,
                  # grails=grails_map,
                  allow_pickle=True)
        
    
    f_name = "./results/test_end.npz"
    np.savez(f_name,  
              I_upper_history=np.array(I_upper_history),
              I_lower_history=np.array(I_lower_history),
              map_reward=map_reward,
              map_spreading=map_spreading,
              map_competence=map_competence,
              grails=grails_map,
              tot_trials=tot_trials,
              tot_iteration=c_iteration,
              hyperparameter=hyperparameter,
              allow_pickle=True)
    
    
    #TODO: beta[0.1,0.5], stop with competence = 1, test every 50 epochs
    
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
