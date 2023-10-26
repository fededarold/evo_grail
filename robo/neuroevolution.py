# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:48:39 2023

@author: darol
"""

import sys
sys.path.insert(0, "C:/Dropbox/jobs/PILLAR/EVO-GRAIL/robo/")

import numpy as np
import torch
import random

import copy

from randomizer import factory_randomizer


class AgentBase():
    '''
    Base Agent Class for Neuroevolution containing the genome and an optional agent id.
    
    '''
    
    def __init__(self, genome_size: int, 
                 random_distribution="normal", genome_data_type="torch"):
                 # **randomizer_parameters): 
        '''
        Parameters
        ----------
        genome_size : int
            Size of the genome
        random_distribution : TYPE, optional
            DESCRIPTION. The default is "normal".
        genome_data_type : TYPE, optional
            DESCRIPTION. The default is "torch".
        randomizer parameters : parameters of the randomizer

        Raises
        ------
        ValueError
            Returns if distribution is not normal or uniform
            or datatype is not python, torch, or numpy

        Returns
        -------
        None.

        '''
        
        # self.random_distribution = random_distribution
        self._genome_size = genome_size
        self._genome_data_type = genome_data_type
        self._random_distribution = random_distribution
        # self._randomizer_parameters = randomizer_parameters["randomizer_parameters"]
        
        if random_distribution != "uniform" and random_distribution != "normal":
            raise ValueError("random initialization must be uniform or normal")
        
        if genome_data_type=="python":
            self.genome = [0]*genome_size
        elif genome_data_type=="numpy":
            self.genome = np.zeros(genome_size)
        elif genome_data_type=="torch":
            self.genome = torch.zeros(genome_size)
        else:
            raise ValueError("genome_data_type can be 'python', 'numpy' or 'torch'")
            
    def set_agent_id(self, agent_id: int):
        self._agent_id = agent_id
        
    # def get_randomizer_parameters(self):
    #     return self._randomizer_parameters


class EvolutionUtils():
    
    @staticmethod
    def set_randomizer(distribution: str,
                        data_type: str,
                        params: dict):
        
        randomizer = factory_randomizer(
            distribution=distribution, 
            data_type=data_type,
            params=params)     
        
        return randomizer
    
    @staticmethod
    def mutate(agent, randomizer):
        
       genome = randomizer.mutate(genome=agent.genome,
                         param_A=randomizer.mutation_param_A,
                         param_B=randomizer.mutation_param_B)
       
       return genome
   
    @staticmethod
    def initialize_genome(agent, randomizer):
        
       genome = randomizer.mutate(genome=agent.genome,
                         param_A=randomizer.initialization_param_A,
                         param_B=randomizer.initialization_param_B)
       
       return genome

# TODO: define a sort of Verbose or Logging variable to save histories
class NeuroEvolution():
    
    def __init__(self, population_size: int, agent: any, 
                 homogeneous_genomes=False, 
                 tot_generations=None, agents_id=False):
        
        base_class = []
        for name in agent.__class__.__bases__:
            base_class.append(name.__name__)
        if agent.__class__.__name__ != "AgentBase" and "AgentBase" not in base_class:
            raise ValueError("a valid agent is an instance of AgentBase class or its child")
        
        self._population_size = population_size
        self._population = np.zeros(population_size, dtype=object)
        self._fitness = np.zeros(population_size)
        self._homogeneous_genomes = homogeneous_genomes
        self._agent_prototype = agent
        self._randomizer_parameters = agent.get_randomizer_parameters()
        
# TODO: define a sort of Verbose or Logging variable to save histories
        self._agents_id = agents_id
        
        self._tot_generations = tot_generations
        if tot_generations is not None:
            self._tot_generations = tot_generations+1
            self._population_history = np.zeros((self._tot_generations,
                                                 self._population_size), 
                                                dtype=object)
        
        
        self._randomizer = EvolutionUtils.set_randomizer(
            self._agent_prototype._random_distribution,
            self._agent_prototype._genome_data_type,
            self._randomizer_parameters)
        self._initialize_population()     

        
    def _initialize_population(self):
        
        for i in range(self._population_size):
            if not self._homogeneous_genomes:
                self._population[i] = copy.deepcopy(self._agent_prototype)
                EvolutionUtils.initialize_genome(self._population[i], self._randomizer)
            else: 
                if i==0:
                    self._population[i] = copy.deepcopy(self._agent_prototype)
                    EvolutionUtils.initialize_genome(self._population[i], self._randomizer)
                else:
                    self._population[i] = copy.deepcopy(self._population[0]) 
                     
# TODO: define a sort of Verbose or Logging variable to save histories   
            if self._agents_id:
                self._population[i].set_agent_id(agent_id=i)
                
# TODO: define a sort of Verbose or Logging variable to save histories
            if self._tot_generations is not None:
                self._generation = 0
                self._population_history[self._generation] = self._population
                self._generation += 1
                     
       
    def selection_rank(self, n_parents: int):
        n_offspring = int((self._population_size-n_parents) / n_parents)
        rank = self._fitness.argsort()[::-1]
        for offspring in range(n_offspring):
            best_agents = copy.deepcopy(self._population[rank[:n_parents]])        
            for agent in best_agents:
                EvolutionUtils.mutate(agent, self._randomizer)                   
            self._population[rank[n_parents*(offspring+1):n_parents*(offspring+2)]] = copy.deepcopy(best_agents)
        self._reset_fitness()

# TODO: define a sort of Verbose or Logging variable to save histories        
        if self._tot_generations is not None:
            self._population_history[self._generation] = self._population 
            self._generation += 1
                     
    
    def _reset_fitness(self):
        self._fitness *= 0
        
            
    def set_fitness(self, agent_id: int, fitness):
        self._fitness[agent_id] = fitness
        
    def get_genome(self, agent_id: int):
        return self._population[agent_id].genome
    
    def get_agents_id(self):
        ids = np.zeros(self._population_size)
        for i in range(self._population_size):
            ids[i] = self._population[i]._agent_id
        return ids
            

# TODO: define a sort of Verbose or Logging variable to save histories    
    def save_evolution(self, file_name: str):
        if self._tot_generations is None:
            np.savez(file=file_name,
                     fitness=self._fitness,
                     population=self._population,
                     pickle=True)
        else:
            np.savez(file=file_name,
                     fitness=self._fitness,
                     population=self._population_history,
                     allow_pickle=True)


class MAPElites():
    def __init__(self, grid_size: int, agent: any, 
                 batch_initial: int, batch_size: int,
                 tot_generations=None, agents_id=False,
                 **randomizer_parameters):
        
        self._grid = np.array((grid_size,grid_size), dtype=object)
        self._grid_segment = 1. / grid_size
        self._fitness = np.array((grid_size,grid_size))
        if tot_generations is not None:
            self._population_history = np.array((grid_size,grid_size))
        self._agent_prototype = agent
        self._randomizer_parameters = randomizer_parameters["randomizer_parameters"]
        self._set_randomizer()
       
    
    def _set_randomizer(self):
        
        self._randomizer = factory_randomizer(
            distribution=self._agent_prototype._random_distribution, 
            module=self._agent_prototype._genome_data_type,
            params=self._randomizer_parameters)   
        
            
    def get_individual_grid(self):
        while True:
            idx=np.random.randint(self._grid.shape[0])
            idy=np.random.randint(self._grid.shape[1])
            agent = copy.deepcopy(self._grid[idx][idy])
            if agent is not None:
                break
            self._randomizer.mutate(genome=agent.genome,
                                    param_A=self._randomizer.initialization_param_A,
                                    param_B=self._randomizer.initialization_param_B)
        return agent
    
    def get_individual_initialized(self):
        agent = copy.deepcopy(self._agent_prototype)
        self._randomizer.mutate(genome=agent.genome,
                                param_A=self._randomizer.initialization_param_A,
                                param_B=self._randomizer.initialization_param_B)
        return agent
        
        
    '''place the guy in the right cell'''
    def place_individual_in_grid(self, agent: any, fitness: float, 
                                 feature_x: float, feature_y: float):
        idx = int(np.floor(feature_x / self._grid_segment))        
        idy = int(np.floor(feature_y / self._grid_segment)) 
        if idx == self._grid.shape[0]:
            idx -= 1
        if idy == self._grid.shape[1]:
            idy -= 1
        
        if self._grid[idx][idy] is not None:
            if fitness > self._fitness[idx][idy]:
                self._grid[idx][idy] = copy.deepcopy(agent)
                self._fitness[idx][idy] = fitness
        else:
            self._grid[idx][idy] = copy.deepcopy(agent)
            self._fitness[idx][idy] = fitness        
    
        

if __name__ == "__main__":    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
      
    uniform = {"initialization": {"lower bound": 0.,
                                  "upper bound": 1.},
                "mutation": {"lower bound": -0.1,
                              "upper bound": 0.1},
                "limits": {"min value": 0., 
                           "max value": 1.}}
    
    
    
    normal = {"initialization": {"mean": 0.,
                                 "std": 0.1},
                "mutation": {"mean": 0.,
                             "std": 0.001}}
    
    
    agent = AgentBase(genome_size=3, random_distribution="uniform", 
                      genome_data_type="python", randomizer_parameters=uniform,)
    
    params = {
        agent._genome_data_type}
    
    evo = NeuroEvolution(population_size=6, agent=agent, agents_id=True)
    for i in range(evo._population_size):
        evo.set_fitness(agent_id=i, fitness=np.random.randint(0,100))
        
    evo.selection_rank(n_parents=2)
    
    for i in range(10):
        evo.selection_rank(n_parents=2)
        for i in range(evo._population_size):
            print(evo._population[i].genome)