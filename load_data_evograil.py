# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:13:43 2023

@author: darol
"""

import numpy as np
# import entropy_estimators.continuous as info
import scipy.stats as ss
# import npeet.entropy_estimators as ee

# folder_name = "C:/data/EVO_GRAIL/"
folder_name = "D:/EVO_GRAIL/"

folder_name += "temp_01_v2/"
# folder_name += "temp_1/"

import matplotlib.pyplot as plt

N_GOALS = 8
N_CONTEXTS = 1

def range_scaler(data, min_bound, max_bound):
    return (data - min_bound) / (max_bound - min_bound)

def entropy(data, n_bins, states_range, normalize=False):
    
    p_data = np.histogram(data, bins=n_bins, range=states_range)
    p_data = p_data[0] / p_data[0].sum()
    e_data = ss.entropy(p_data) 
    
    if normalize == True:
        e_data /= np.log(n_bins) 
    
    return e_data


def entropy_multivariate(data, n_bins, states_range, normalize=False):
    
    p_data = np.histogramdd(sample=data, bins=n_bins, range=states_range)
    p_data = p_data[0] / p_data[0].sum()
    e_data = ss.entropy(p_data.flatten())
    
    if normalize == True:
        e_data /= np.log(np.prod(n_bins))
    
    return e_data
    

def joint_entropy(data_x, data_y, n_bins, states_range):
    
    p_data_x__data_y = np.histogram2d(data_x, data_y,
                                      bins=n_bins, range=states_range)
    p_data_x__data_y = p_data_x__data_y[0] / p_data_x__data_y[0].sum()
    p_data_x__data_y = p_data_x__data_y.flatten()
    e_data_x__data_y = ss.entropy(p_data_x__data_y)
    
    return e_data_x__data_y


def mutual_information(E_x, E_y, E_xy, normalization=None):
    
    I = E_x + E_y - E_xy
    if normalization == "R":
        I = I / (E_x + E_y)
    elif normalization == "IQR":
        I = I / E_xy

    return I

# def concatenate_data(data, keys, first_loop=None):
    


first_loop = True
# for i in range(1,1502,50):
for i in range(1,52,50):
    
    print(i)   
    
    data = np.load(folder_name + "test_data_" + str(i) + ".npz", allow_pickle=True)
    keys = list(data.keys())
        
    competence_raw = data["goal_competence"] # nested dictionary
    dict_keys_goals = list(competence_raw[0].keys())
    dict_keys_contexts = list(competence_raw[0][dict_keys_goals[0]].keys())
    competence_data = []
    for i in range(len(competence_raw)):
        competence_trial = []
        for j in range(N_GOALS):
            competence_trial.append(competence_raw[i][dict_keys_goals[j]][dict_keys_contexts[0]])
        competence_data.append(competence_trial)
    if first_loop:
        first_loop = False
        action = np.expand_dims(data["action_data"], axis=1)
        competence = np.array(competence_data)        
        sensor_simulator = data["sensor_simulator"] #NO NEED
        sensor_episode_t = data["sensor_episode_t"] #2D array OK
        sensor_episode_t1 = data["sensor_episode_t1"] #2D array OK
        motivation = np.expand_dims(data["motivation_type"], axis=1)
        goal_active = np.expand_dims(data["goal_active"], axis=1)
        goal_state = data["goal_state"]
        # reward = np.expand_dims(data["reward"], axis=1)
        epoch = np.expand_dims(data["epoch"], axis=1)
        trial = np.expand_dims(data["trial"], axis=1)
    else:
        action = np.vstack((action, np.expand_dims(data["action_data"], axis=1)))
        competence = np.vstack((competence, np.array(competence_data)))        
        sensor_simulator = np.vstack((sensor_simulator, data["sensor_simulator"])) #NO NEED
        sensor_episode_t = np.vstack((sensor_episode_t, data["sensor_episode_t"])) #2D array OK
        sensor_episode_t1 = np.vstack((sensor_episode_t1, data["sensor_episode_t1"])) #2D array OK
        motivation = np.vstack((motivation, np.expand_dims(data["motivation_type"], axis=1)))
        goal_active = np.vstack((goal_active, np.expand_dims(data["goal_active"], axis=1)))
        goal_state = np.vstack((goal_state, data["goal_state"]))
        # reward = np.vstack((reward, np.expand_dims(data["reward"], axis=1)))
        epoch = np.vstack((epoch, np.expand_dims(data["epoch"], axis=1)))
        trial = np.vstack((trial, np.expand_dims(data["trial"], axis=1)))


# we do not know the action space and we infer the bounds from data
action = range_scaler(action, np.floor(action.min()), np.ceil(action.max()))


# print(entropy(np.array(sensor_episode_t), n_bins=10, states_range=[0,1]))
# print(ee.entropy(np.array(sensor_episode_t), k =3))


E_competence_multivariate = []
E_competence = []
mean_competence = []
std_competence = []
E_sensor_t = []
E_sensor_t1 = []
E_sensor_t_multivariate = []
E_sensor_t1_multivariate = []
E_action = []

I_sensor_t_t1 = []
I_action_sensor = []
I_sensor_t_action = []
# TODO: do we select the goal at the beginning of the trial?

epoch = epoch.squeeze()
trial = trial.squeeze()
range_data = [(0.,1.)] # * sensor_episode_t.shape[1]
range_data = [(0.,1.)] # * competence.shape[1]
n_bins_multivariate = [10] # * sensor_episode_t.shape[1]
n_bins = 10
for i in range(1,epoch.max()):
    if i%10==0:
        print(i)
    
    # E_competence_bins = []
    # E_sensor_t_bins = []
    # E_sensor_t1_bins = []
    # E_action_bins = []
    data_competence = competence[np.where(epoch==i)[0]]
    mean_competence.append(data_competence.mean())
    std_competence.append(data_competence.std())          
    E_competence.append(entropy(data=data_competence.flatten(),
                                n_bins=n_bins,
                                states_range=range_data[0],
                                normalize=True))
    
    data_sensor_t = sensor_episode_t[np.where(epoch==i)[0]]
    E_sensor_t.append(entropy(data=data_sensor_t.flatten(), 
                              n_bins=n_bins, 
                              states_range=range_data[0], 
                              normalize=True))
    
    data_sensor_t1 = sensor_episode_t1[np.where(epoch==i)[0]]
    E_sensor_t1.append(entropy(data=data_sensor_t1.flatten(), 
                               n_bins=n_bins, 
                               states_range=range_data[0], 
                               normalize=True))

    data_action = action[np.where(epoch==i)[0]]
    E_action.append(entropy(data=data_action.flatten(), 
                                 n_bins=n_bins, 
                                 states_range=range_data[0],
                                 normalize=True))
    
    epoch_trial = trial[np.where(epoch==i)[0]]
    
    I_sensors = 0
    I_t_action = 0 
    I_action_t1 = 0
    n_trials = np.unique(epoch_trial).shape[0]
    for t in range(n_trials):
        
        E_sensor_t_trial = entropy(data=data_sensor_t[np.where(epoch_trial==t)[0]].flatten(), 
                                   n_bins=n_bins, 
                                   states_range=range_data[0])
        
        E_sensor_t1_trial = entropy(data=data_sensor_t1[np.where(epoch_trial==t)[0]].flatten(), 
                                    n_bins=n_bins, 
                                    states_range=range_data[0])
        
        E_action_trial = entropy(data=data_action[np.where(epoch_trial==t)[0]].flatten(), 
                                    n_bins=n_bins, 
                                    states_range=range_data[0])
        
        E_sensor_t_t1_trial = joint_entropy(data_x=data_sensor_t[np.where(epoch_trial==t)[0]].flatten(), 
                                            data_y=data_sensor_t1[np.where(epoch_trial==t)[0]].flatten(), 
                                            n_bins=[n_bins]*2, 
                                            states_range=range_data*2)
        
        
        
        data_action_trial=data_action[np.where(epoch_trial==t)[0]]
        data_action_adjusted = []
        for act in range(len(data_action_trial)):
            data_action_adjusted.append([data_action_trial[act][0]]*6)
        data_action_adjusted = np.array(data_action_adjusted).flatten()
        # data_action_adjusted = np.expand_dims(data_action_adjusted, axis=1)
        
        
        E_sensor_t_action_trial = joint_entropy(data_x=data_sensor_t[np.where(epoch_trial==t)[0]].flatten(), 
                                                data_y=data_action_adjusted.flatten(), 
                                                n_bins=[n_bins]*2, 
                                                states_range=range_data*2)
        
        E_action_sensor_t1_trial = joint_entropy(data_x=data_action_adjusted.flatten(), 
                                                 data_y=data_sensor_t1[np.where(epoch_trial==t)[0]].flatten(), 
                                                 n_bins=[n_bins]*2, 
                                                 states_range=range_data*2)
        
        I_sensors += mutual_information(E_x=E_sensor_t_trial, E_y=E_sensor_t1_trial, E_xy=E_sensor_t_t1_trial)
        I_t_action += mutual_information(E_x=E_sensor_t_trial, E_y=E_action_trial, E_xy=E_sensor_t_action_trial)
        I_action_t1 += mutual_information(E_x=E_action_trial, E_y=E_sensor_t1_trial, E_xy=E_action_sensor_t1_trial)
    
    I_sensor_t_t1.append(I_sensors / n_trials)
    I_sensor_t_action.append(I_t_action / n_trials)
    I_action_sensor.append(I_action_t1 / n_trials)
    # E_competence.append(E_competence_bins)
    # E_sensor_t.append(E_sensor_t_bins)
    # E_sensor_t1.append(E_sensor_t1_bins)
    # E_action.append(E_action_bins)
        
print(folder_name)

# goal_epoch = []
# for i in range(1,epoch.max()+1):
#     tmp_goal_active = goal_active[np.where(epoch==i)[0]]
#     goal_epoch.append(tmp_goal_active)
    



# E_competence_multivariate.append(entropy_multivariate(data=tmp_competence, 
#                                  n_bins=n_bins_multivariate*competence.shape[1], 
#                                  states_range=range_data*competence.shape[1], 
#                                  normalize=True))

# E_sensor_t_multivariate.append(entropy_multivariate(data=sensor_episode_t[np.where(epoch==i)[0]], 
#                                n_bins=n_bins_multivariate*sensor_episode_t.shape[1], 
#                                states_range=range_data*sensor_episode_t.shape[1], 
#                                normalize=True))

# E_sensor_t1_multivariate.append(entropy_multivariate(data=sensor_episode_t1[np.where(epoch==i)[0]], 
#                                 n_bins=n_bins_multivariate*sensor_episode_t1.shape[1], 
#                                 states_range=range_data*sensor_episode_t1.shape[1], 
#                                 normalize=True))
