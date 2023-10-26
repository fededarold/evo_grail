# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:20:52 2023

@author: darol
"""

# from __future__ import annotations
# from abc import ABC, abstractmethod

import numpy 
import torch
import random


def factory_randomizer(distribution: str, data_type: str, **params):
    '''
    Factory function for randomizers.
    This is a wrapper to use vanilla python, torch, and numpy 
    with numpy syntax.

    Parameters
    ----------
    distribution : str
        The random sampling. Normal or uniform.
    data_type : str
        Python, torch, or numpy
    **params : TYPE
        Parameters for the distrubion.

    Returns
    -------
    A randomizer class
        

    '''
    params = params["params"]
    randomizer = {
        "torch": {"uniform" : _UniformPytorch,
                  "normal" : _NormalPytorch},
        "numpy": {"uniform" : _UniformNumpy,
                  "normal" : _NormalNumpy},
        "python": {"uniform" : _UniformPython,
                   "normal" : _NormalPython}}
    
    return randomizer[data_type][distribution](params=params)
    
    


class _BaseClass():
    '''
    Base randomizer class storing 2 parameters for initializing the genome
    and 2 parameters for sampling the mutations
    
    '''
    def __init__(self, initialization_param_A: float, initialization_param_B: float,
                 mutation_param_A: float, mutation_param_B: float):
        self.initialization_param_A = initialization_param_A
        self.initialization_param_B = initialization_param_B
        self.mutation_param_A = mutation_param_A
        self.mutation_param_B = mutation_param_B



class _UniformBaseClass(_BaseClass):
    def __init__(self, **params):
        '''
        Parameters
        ----------
        **params : 
            range min and max 
            upper and lower bound for the mutations

        Returns
        -------
        None.
        Parameters are 
        
        '''
        params = params["params"]
        super(_UniformBaseClass, self).__init__(
            initialization_param_A=params["initialization"]["lower bound"],
            initialization_param_B=params["initialization"]["upper bound"],
            mutation_param_A=params["mutation"]["lower bound"],
            mutation_param_B=params["mutation"]["upper bound"])        
        self._min_value = params["limits"]["min value"]
        self._max_value = params["limits"]["max value"]     
        

class _NormalBaseClass(_BaseClass):
    def __init__(self, **params):
        '''
        Parameters
        ----------
        **params : 
            mean and std

        Returns
        -------
        None.

        '''
        
        params = params["params"]
        super(_NormalBaseClass, self).__init__(
            initialization_param_A=params["initialization"]["mean"],
            initialization_param_B=params["initialization"]["std"],
            mutation_param_A=params["mutation"]["mean"],
            mutation_param_B=params["mutation"]["std"])           



# TODO: second allele is not working
class _UniformPython(_UniformBaseClass):
    def __init__(self, **params): 
        params = params["params"]
        super(_UniformPython, self).__init__(params=params) 
    
    def mutate(self, genome: list, param_A: float, param_B: float):
        for i in range(len(genome)):
            genome[i] += random.uniform(a=param_A, b=param_B)
        #clamp
        idx = [idx for idx, val in enumerate(genome) if val < self._min_value]
        for i in range(len(idx)):
            genome[idx[i]] = self._min_value
        idx = [idx for idx, val in enumerate(genome) if val > self._max_value]
        for i in range(len(idx)):
            genome[idx[i]] = self._max_value        
        return genome
    
    
class _UniformNumpy(_UniformBaseClass):
    def __init__(self, **params): 
        params = params["params"]
        super(_UniformNumpy, self).__init__(params=params) 
    
    def mutate(self, genome: numpy.ndarray, param_A: float, param_B: float):
        genome += numpy.random.uniform(low=param_A, high=param_B, size=genome.size)
        #clamp
        return genome.clip(min=self._min_value, max=self._max_value)


class _UniformPytorch(_UniformBaseClass):
    def __init__(self, **params): 
        params = params["params"]
        super(_UniformPytorch, self).__init__(params=params)    
       
    def mutate(self, genome: torch.Tensor, param_A: float, param_B: float):
        # we get first dimension [0] of rand as genome is a vecotr and rand is a 2D Tensor
        genome += (param_B - param_A) * torch.rand(size=(1,genome.shape[0]))[0] + param_A
        return genome.clamp(self._min_value, self._max_value)


 

class _NormalPython(_NormalBaseClass):
    def __init__(self, **params):
        params = params["params"]
        super(_NormalPython, self).__init__(params=params)
        
    def mutate(self, genome: list, param_A: float, param_B: float):
        for i in range(len(genome)):
            genome[i] += random.gauss(mu=param_A, sigma=param_B)
        return genome


class _NormalNumpy(_NormalBaseClass):
    def __init__(self, **params):
        params = params["params"]
        super(_NormalNumpy, self).__init__(params=params)
    def mutate(self, genome: numpy.ndarray, param_A: float, param_B: float):
        genome += numpy.random.normal(loc=param_A, scale=param_B, size=genome.size)
        return genome

    

class _NormalPytorch(_NormalBaseClass): 
    def __init__(self, **params):
        params = params["params"]
        super(_NormalPytorch, self).__init__(params=params)
    
    def mutate(self, genome: torch.Tensor, param_A: float, param_B: float):
        # we get first dimension [0] of rand as genome is a vecotr and rand is a 2D Tensor
        genome += torch.normal(mean=param_A, std=param_B, size=(1,genome.shape[0]))[0]
        return genome
        
