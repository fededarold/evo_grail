# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:21:31 2021

@author: Fede Laptop
"""

########## IMPORTS ########### {{{1
# from gym.envs.box2d import BipedalWalker

from MLP_fede import MLP
from individual_ga_fede_evomutation import Individual
from container_fede import Container


import random
import torch
import numpy as np

import time
import yaml
import argparse

from experiment_population import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_gen", default=100, type=int, help="starting generation")
    parser.add_argument("--timestamp", type=str, help = "experiment timestamp id")
    parser.add_argument("--individual_id", type=int, help = "id of best individual")
    parser.add_argument("--dir", default="./", type=str,  help ='data directory')
    
    args = parser.parse_args()
   
    parameters = np.load(args.dir + "/" + args.timestamp + "_params.npz", allow_pickle=True)["arr_0"].item()
    
    exp_test = Experiment(params=parameters, experiment_timestamp=args.timestamp, folder=args.dir, starting_generation=args.start_gen)
    
    if args.start_gen == 0:
        best_individuals = exp_test.initialize()
        best = best_individuals[args.individual_id]
    else:
        best = None
    # for i in range(len(best_individuals)):
    exp_test.run_evolution(individual_id=str(args.individual_id), 
                           individual_best=best)
    
if __name__ == "__main__":
    main()