# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 21:52:00 2023

@author: darol
"""


import numpy as np
# import entropy_estimators.continuous as info
import scipy.stats as ss
# import npeet.entropy_estimators as ee

class InfoDiscrete:
    
    @staticmethod
    def range_scaler(data, min_bound, max_bound):
        return (data - min_bound) / (max_bound - min_bound)
    
    @staticmethod
    def entropy(data, n_bins, states_range, normalize=False):
        
        p_data = np.histogram(data, bins=n_bins, range=states_range)
        p_data = p_data[0] / p_data[0].sum()
        e_data = ss.entropy(p_data) 
        
        if normalize == True:
            e_data /= np.log(n_bins) 
        
        return e_data
    
    @staticmethod
    def entropy_multivariate(data, n_bins, states_range, normalize=False):
        
        p_data = np.histogramdd(sample=data, bins=n_bins, range=states_range)
        p_data = p_data[0] / p_data[0].sum()
        e_data = ss.entropy(p_data.flatten())
        
        if normalize == True:
            e_data /= np.log(np.prod(n_bins))
        
        return e_data
        
    @staticmethod
    def joint_entropy(data_x, data_y, n_bins, states_range):
        
        p_data_x__data_y = np.histogram2d(data_x, data_y,
                                          bins=n_bins, range=states_range)
        p_data_x__data_y = p_data_x__data_y[0] / p_data_x__data_y[0].sum()
        p_data_x__data_y = p_data_x__data_y.flatten()
        e_data_x__data_y = ss.entropy(p_data_x__data_y)
        
        return e_data_x__data_y
    
    @staticmethod
    def mutual_information(E_x, E_y, E_xy, normalization=None):
        
        I = E_x + E_y - E_xy
        if normalization == "R":
            I = I / (E_x + E_y)
        elif normalization == "IQR":
            I = I / E_xy
    
        return I

# def concatenate_data(data, keys, first_loop=None):
    


