# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:06:44 2022

@author: darol
"""


from gym.envs.box2d import BipedalWalker, BipedalWalkerHardcore, LunarLander, LunarLanderContinuous
import gym

from MLP_fede import MLP
from individual_ga_fede_evomutation_v2 import Individual
from population_evolution import Population
from fede_robustness_all_v2 import Robustness

import random
import torch
import numpy as np

import time

import yaml

import copy
import os

class Experiment:
    def __init__(self, params, 
                 experiment_timestamp: str,
                 folder: str,
                 starting_generation=0,
                 population_size = 100, generations=500):
        
        
        # self.base_folder = "D:/post_gecco/results/"
        self.base_folder = folder
        self.starting_generation = starting_generation
        self.params = params
        
        self.params["save_every_n_generations"] = 100
        
        random.seed(self.params["seed"])
        torch.manual_seed(self.params["seed"])
        np.random.seed(self.params["seed"])
        
        self.timestamp = experiment_timestamp
        self.dir_name = self.base_folder + "/" + experiment_timestamp
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        self.file_name = self.base_folder + "/" + self.timestamp + "_s_" + str(params["seed"]) + "_g_"
        # '''save parameters'''
        # np.savez(self.base_folder + params["folder_name"] + "/" + self.timestamp + '_params', self.params, allow_pickle=True)
        
        self.population_size = population_size
        self.population = Population(population_size, generations)
        
        # self.initialize()  
    
    
    def select_environment(self):
        env = None
        if self.params["environment"] == "BipedalWalker":
            env = BipedalWalker()
        elif self.params["environment"] == "BipedalWalkerHardcore":
            env = BipedalWalkerHardcore()
        elif self.params["environment"] == "LunarLander":
            env = LunarLander()
        elif self.params["environment"] == "LunarLanderContinuous":
            env = gym.make("LunarLander-v2", continuous=True)
        elif self.params['environment'] == 'MountainCar':
            env = gym.make('MountainCarContinuous-v0')
        env.reset(seed=self.params["seed"])
        
        return env
    
    
    def initialize(self, make_directories=True):
        """load the first grid of the MAP-ELITES experiment"""
        file_load = self.base_folder + \
            "/" + self.timestamp + "_s_" + str(self.params["seed"]) + \
            "_g_" + str(self.params["save_every_n_generations"]) + ".npz"
        data = np.load(file_load, allow_pickle=(True))
        '''load grid'''
        initial_grid = data['arr_2'][0]
        data.close()
        
        file_load = self.base_folder + \
            "/" + self.timestamp + '_grid_history.npz'
        data = np.load(file_load, allow_pickle=(True))
        individuals_solved = data["arr_12"]
        spreading_grid = data["arr_18"][0]
        data.close()
        
        '''find bests'''
        best_individuals = []
        for i in range(len(individuals_solved)):
            idx = np.where(spreading_grid == individuals_solved[i])
            best_individuals.append(initial_grid[idx[0].item()][idx[1].item()])
            if make_directories:
                if not os.path.exists(self.dir_name + "/" + str(i)):
                    os.mkdir(self.dir_name + "/" + str(i))
                
            
        return best_individuals
        
    
    def run_evolution(self, individual_id: str, individual_best=None):
        
        dir_name_individual = self.dir_name + "/" + individual_id + "/" + self.timestamp + "_s_" + str(self.params["seed"]) + "_g_"
 
        
        print("seed" + str(self.params["seed"]))
        print(self.params["environment"] + " population")
                
        
        # # env = BipedalWalker()
        # env = BipedalWalkerHardcore()
        env = self.select_environment()
        # env.seed(self.params["seed"])
        
        mlp_test = MLP(self.params["n_inputs"], self.params["n_hiddens"], 
                       self.params["n_outputs"], self.params["is_mask"])
        # container_test = Container(self.params["container_x"], self.params["container_y"], mlp_test)
        
        '''initialize the population'''
        population = Population()
        if self.starting_generation == 0:        
            population.initialize_population(individual_best=individual_best)
        else:
            population._population_vector = np.load(dir_name_individual + str(self.starting_generation) + ".npz", 
                                                    allow_pickle=True)["arr_1"][-1]
        
        '''save fitness'''
        #fitness_grid = np.empty((self.params["generations"], self.params["container_x"], self.params["container_y"]))
        fitness_grid = np.empty((self.params["save_every_n_generations"], self.population_size))
        fitness_grid[:] = np.nan
        
        '''saving all grids every n generations'''
        grid_all = np.empty((self.params["save_every_n_generations"]), dtype=object)
        grid_all[:] = np.nan
        
        # '''debugging grid selection'''
        # selection_debug = np.zeros((self.params["generations"], self.params["container_x"], self.params["container_y"]))
        # # selection_debug[:] = np.nan
            
        
        '''evolve'''
        for g in range(self.starting_generation, population._tot_generations):
            
            # fitness_generation_vec = np.zeros(population._population_size)
            # if g%10==0:
            print(g)
                        
            for i in range(population._population_size):
                
                individual = population.get_individual(i)                
                individual.reset_fitness()
                mlp_test.set_model_params(individual.genome_weights, individual.genome_masks)
                
                individual_fitness = np.zeros(self.params["trials"])
                for t in range(self.params["trials"]): 
                    env.reset()
                    done = False
                    '''we do the first step with X=0 for now'''
                    output = mlp_test.forward(torch.zeros(self.params["n_inputs"]))
                    for _ in range(self.params["world_iterations"]):
                        
                        # env.render('human')
                        if self.params["environment"] == "LunarLander":
                            output = output.argmax()
                            
                        obs, reward, terminated, truncated, _ = env.step(output)
                        if terminated or truncated:
                            done=True
                        output = mlp_test.forward(obs)
                        individual_fitness[t] += reward
                       
                        if done:
                            break
                                
                fitness_mean = np.mean(individual_fitness)
                individual.set_fitness(fitness_mean)
                population.set_fitness(i, fitness_mean)
                
            population.reproduce()
                      
                      
            #print(container_test.grid)  
            if g%self.params["save_every_n_generations"] == 0 and g > 0:                 
                c_file = dir_name_individual + str(g)
                np.savez(c_file, fitness_grid, grid_all, allow_pickle=True)
                grid_all = np.empty((self.params["save_every_n_generations"]),dtype=object)
                grid_all[:] = np.nan   
                #fitness_grid = np.empty((self.params["generations"], self.params["container_x"], self.params["container_y"]))
                fitness_grid = np.empty((self.params["save_every_n_generations"], self.population_size))
                fitness_grid[:] = np.nan
                #spreading_grid = np.empty((self.params["generations"], self.params["container_x"], self.params["container_y"]))
                
            '''store data'''
            grid_all[g%self.params["save_every_n_generations"]] = population.get_population_vector()   
            fitness_grid[g%self.params["save_every_n_generations"]] = population.get_fitness_vector()
            
                        
            if g%self.params["print_every_n_generations"] == 0:
                # print(g)
                print(np.nanmax(fitness_grid[g%self.params["save_every_n_generations"]]))
                
        #grid = container_test.get_grid()   
        print(np.nanmax(fitness_grid[g%self.params["save_every_n_generations"]])) 
        '''below use this trick as we import data every n timestep'''
        c_file = dir_name_individual + str(g + (self.params["save_every_n_generations"] - g%self.params["save_every_n_generations"]))
        np.savez(c_file, fitness_grid, grid_all, allow_pickle=True)
        env.close()
        
        # '''do robustness test'''
        # r_path = self.base_folder + self.params["folder_name"] + "/"
        # generalization_test = Robustness(n_trials=self.params["robustness_trials"], 
        #                                  path=r_path, f_name=self.timestamp,
        #                                  model=mlp_test, grid=container_test.get_grid(),
        #                                  params=self.params, env=env)
        # generalization_test.run_test()
        
