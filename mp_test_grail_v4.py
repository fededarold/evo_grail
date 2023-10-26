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

uniform = {"initialization": {"lower bound": 0.01,
                              "upper bound": 0.99},
            "mutation": {"lower bound": -0.01,
                          "upper bound": 0.01},
            "limits": {"min value": 0.01, 
                       "max value": 0.99}}


class Dummy:
    def __init__(self, name, method):
        self.name = name
        self.method = method

    def get(self, *args, **kwargs):
        return self.method(self.name, args, kwargs)


class ObjProxy(NamespaceProxy):
    """Returns a proxy instance for any user defined data-type. The proxy instance will have the namespace and
    functions of the data-type (except private/protected callables/attributes). Furthermore, the proxy will be
    pickable and can its state can be shared among different processes. """

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, types.MethodType):
            return Dummy(name, self._callmethod).get
        return result

class AgentGRAIL(ga.AgentBase):
    
    def __init__(self):
        super(AgentGRAIL, self).__init__(genome_size=1, genome_data_type="python", random_distribution="uniform")
        self.grail = MOTIVEN()
        # self._genome_size = 1
        # self._genome_data_type = "python"
        # self._random_distribution = "uniform"   
        
        
    def run_agent(self):
        self.grail.main()
        
    @classmethod
    def create(cls, *args, **kwargs):
        # Register class
        class_str = cls.__name__
        BaseManager.register(class_str, cls, ObjProxy, exposed=tuple(dir(cls)))
        # Start a manager process
        manager = BaseManager()
        manager.start()

        # Create and return this proxy instance. Using this proxy allows sharing of state between processes.
        inst = eval("manager.{}(*args, **kwargs)".format(class_str))
        return inst

def main_fun(G, current_epoch: int, epoch_increase=10): #, iterations=10):
    
    N=len(G)
    max_epoch = current_epoch + epoch_increase
    for i in range(N):
        G[i].max_epochs = max_epoch
        
        
    with Pool(N) as pool:     
        
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

# self.x = (feature_x_max - feature_x_min) / self.grid.shape[0]
# self.y = (feature_y_max - feature_y_min) / self.grid.shape[1]


# def place_individuals_in_map(new_individuals, map_grid,
#                              adaptive_x=False, adaptive_y=False):

#     grid_x = map_grid.shape[0] 
#     grid_y = map_grid.shape[1]
    
#     for i in range(len(individuals)):
#         x, y = get_individual_xy_grid(feature_x, feature_y, feature_x_min, feature_x_max, feature_y_min, feature_y_max)


#def main_fun(tot_epochs: int, epoch_increase=1, iterations=10):
if __name__ == "__main__":
    
    set_start_method("spawn")
    
    # Tlist = [0.1, 0.01, 0.9, 0.05, 0.1] #, 0.05, 0.01]
    # # iterations = [100, 1000, 100000]
    # N = len(Tlist)
    # G = []
    
    TEST_EVERY_INITIAL_TRAINING = 10
    N_ITERATIONS = 10
    c_iteration = 0
    TEST_EVERY = 10
 
    t_0 = time.time()
    
    random.seed(1)
    np.random.seed(1)

    n_agents = 10
    uniform = {"initialization": {"lower bound": 0.01,
                                  "upper bound": 0.99},
                "mutation": {"lower bound": -0.1,
                              "upper bound": 0.1},
                "limits": {"min value": 0.01, 
                           "max value": 0.99}}
    
    
    I_upper_history = []
    I_lower_history = []
    
    randomizer = rng.factory_randomizer(distribution="uniform", 
                                        data_type="python", 
                                        params=uniform)

    genome = [0]
    genome_list = []
    for i in range(n_agents):
        genome_list.append(randomizer.mutate(genome=copy.deepcopy(genome),
                                             param_A=randomizer.initialization_param_A, 
                                             param_B=randomizer.initialization_param_B))
    
    grails_map_parents = np.empty((10,10), dtype=object)
    # grails_map_offspring = np.empty((10,10), dtype=object)
    map_spreading = np.zeros((N_ITERATIONS,10,10))
    map_spreading[:] = np.nan
    map_reward = np.zeros((N_ITERATIONS,10,10))
    map_reward[:] = np.nan
    
    grails_initial_batch = np.empty((n_agents), dtype=object)
    for i in range(n_agents):
        grails_initial_batch[i] = MOTIVEN.create()
        grails_initial_batch[i].set_softmax_temperature(genome_list[i])
        grails_initial_batch[i].agent_id = i
        grails_initial_batch[i].init_main()
        print("init grail: " + str(i))
    print("done")    
    
    
    grails_initial_batch, current_epoch = main_fun(grails_initial_batch, 
                                                   current_epoch=0,     
                                                   epoch_increase=TEST_EVERY_INITIAL_TRAINING) 
    
    #get min-max values of the necessary BD dimensions
    I_values = []
    for i in range(n_agents):
        I_values.append(grails_initial_batch[i].I_sensors_action)
    I_upper_bound = max(I_values)
    I_lower_bound = min(I_values)
    
    I_upper_history.append(I_upper_bound)
    I_lower_history.append(I_lower_bound)
    
    for i in range(n_agents):
        I_scaled = InfoDiscrete.range_scaler(grails_initial_batch[i].I_sensors_action,
                                             I_lower_bound, I_upper_bound)
        
        x, y = get_individual_xy_grid(feature_x=grails_initial_batch[i].entropy_goal, 
                                      feature_y=I_scaled)
        
        if grails_map_parents[x][y] is None:
            grails_map_parents[x][y] = grails_initial_batch[i]
        else:
            if grails_map_parents[x][y].reward_test < grails_initial_batch[i].reward_test:
               grails_map_parents[x][y] = grails_initial_batch[i]
            elif grails_map_parents[x][y].reward == grails_initial_batch[i].reward:
                if np.random.rand() > 0.5:
                    grails_map_parents[x][y] = grails_initial_batch[i]
    
    
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
                individual += 1
                
    # c_iteration += 1
    
    np.savez("test_save_init.npz", 
              I_upper_history=np.array(I_upper_history),
              I_lower_history=np.array(I_lower_history),
              map_reward=map_reward,
              map_spreading=map_spreading,
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
                        
        # assess parents
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
                    individual += 1
        
        f_name = "test_" + str(c_iteration) + ".npz"
        np.savez("test_save_init.npz", 
                  I_upper_history=np.array(I_upper_history),
                  I_lower_history=np.array(I_lower_history),
                  map_reward=map_reward,
                  map_spreading=map_spreading,
                  allow_pickle=True)
