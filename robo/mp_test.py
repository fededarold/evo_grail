# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:53:56 2023

@author: darol
"""

from multiprocessing import Pool
from multiprocessing.managers import NamespaceProxy, BaseManager
import types
import numpy as np
import time


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


class monte_carlo:

    def __init__(self, ): #params):
        
        # self.T = params["T"]
        # self.iterations = params["I"]
        
        # self.set_init_parameters(params)
        
        self.x = np.ones((1000, 3))
        self.E = np.mean(self.x)
        self.Elist = []
        self.T = None

    def simulation(self, ): #temperature, iterations):
        
        t = time.time()
        # self.T = temperature
        for i in range(self.iterations):
            self.MC_step()
            if i % 10 == 0:
                self.Elist.append(self.E)
        
        # print(iterations, flush=True)
        # print(time.time() - t, flush=True)
        # print("\n")
        
        return

    def MC_step(self):
        x = self.x.copy()
        k = np.random.randint(1000)
        x[k] = (x[k] + np.random.uniform(-1, 1, 3))
        temp_E = np.mean(self.x)
        if np.random.random() < np.exp((self.E - temp_E) / self.T):
            self.E = temp_E
            self.x = x
        return temp_E
    
    def set_init_parameters(self, params):
        self.T = params["T"]
        self.iterations = params["I"]

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


def proba(dE,pT):
    return np.exp(-dE/pT)



if __name__ == "__main__":
    Tlist = [1.1, 1.2, 1.3]
    iterations = [100, 1000, 100000]
    N = 2 #len(Tlist)
    G = []
    
    params_list = [
        {"T":1.1, "I":100},
        {"T":1.3, "I":10000}
        ]

    # Create our managed instances
    for i in range(N):
        G.append(monte_carlo.create())
        G[i].set_init_parameters(params_list[i])

    for _ in range(5):

        #  Run simulations in the manager server
        results = []
        with Pool(N) as pool:

            for i in range(N):  # this loop should be ran in multiprocess
                results.append(pool.apply_async(G[i].simulation, )) #(Tlist[i], iterations[i])))

            # Wait for the simulations to complete            
            for result in results:
                result.get()
    
            # print("done")

        for i in range(N // 2):
            
            # print("loop")
            
            dE = G[i].E - G[i + 1].E
            pT = G[i].T + G[i + 1].T
            p = proba(dE, pT)  # (proba is a function, giving a probability depending on dE)
            if np.random.random() < p:
                T_temp = Tlist[i]
                Tlist[i] = Tlist[i + 1]
                Tlist[i + 1] = T_temp

    print(Tlist)