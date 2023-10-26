# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:53:56 2023

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

from grail.motiven_bandit_ale_MP import MOTIVEN

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


def main_fun(tot_epochs: int, epoch_increase=1, iterations=10):
    Tlist = [0.1, 0.01, 0.9, 0.05, 0.1] #, 0.05, 0.01]
    # iterations = [100, 1000, 100000]
    N = len(Tlist)
    G = []
 
    
    for i in range(N):
        G.append(MOTIVEN.create())
        G[i].set_softmax_temperature(Tlist[i])
        G[i].init_main()
        
    t0_learn = time.time()
      
    with Pool(N) as pool:       
        print("test")
        for batch in range(iterations):
            
        # for epoch in range(int(tot_epochs/eval_every_n_epoch)):
        #     print(epoch)
    # pool = Pool(processes=2)        
    
        # print(str(time.time()-t0))
        
            for i in range(N):
                G[i].max_epochs = tot_epochs
            
            results = []
            for i in range(N):  # this loop should be ran in multiprocess
                # results.append(pool.apply_async(G[i].run_agent, )) #(Tlist[i], iterations[i])))
                # results.append(pool.apply_async(G[i].init_main, ))
                results.append(pool.apply_async(G[i].main, ))
                # results.append(pool.apply_async(G[i].get_data, ))
                
            # Wait for the simulations to complete            
            for result in results:
                result.get()
                
            # pool.close()
            # pool.join()
            
            print(G[0].n_epochs)
            tot_epochs += epoch_increase
    
    print("learn: " + str(time.time()-t0_learn))   
    
    t0_test = time.time()
            
    for i in range(N):   
        G[i].get_data()
        # action_data_1[i].append(G[i].action_data)
        # sensors_data_1[i].append(G[i].sensors_data)
        # goal_data_1[i].append(G[i].goal_data)
    
    print("learn: " + str(time.time()-t0_test)) 
    
    pool.close()
    pool.join()
    
    return G, tot_epochs
    

if __name__ == "__main__":
    
    G, tot_epochs = main_fun(tot_epochs=0) 
    for i in range(3):
        G, tot_epochs = main_fun(tot_epochs=tot_epochs)
                
                
        # print(G[0].n_epochs)
        # tot_epochs += epoch_increase
            
            
    
            
            # print("data")
            
            # results = []
            # for i in range(N):  # this loop should be ran in multiprocess
            #     # results.append(pool.apply_async(G[i].run_agent, )) #(Tlist[i], iterations[i])))
            #     # results.append(pool.apply_async(G[i].init_main, ))
            #     results.append(pool.apply_async(G[i].get_data, ))
            #     # results.append(pool.apply_async(G[i].get_data, ))
                
            # # Wait for the simulations to complete            
            # for result in results:
            #     result.get()
        
            
    
            
            # action_data_2.append(G[1].action_data)
            # sensors_data_2.append(G[1].sensors_data)
            # goal_data_2.append(G[1].goal_data)
        
    # pool.close()
    # pool.join()
    #     # 
            
 

# TODO: use parameters in main: get the current epoch from G[0] and pass it
