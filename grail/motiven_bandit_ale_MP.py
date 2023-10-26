import sys
sys.path.insert(0, "C:/Dropbox/jobs/PILLAR/EVO-GRAIL/grail/")

import pickle

import numpy as np
from numpy.matlib import repmat 
from matplotlib import pyplot as plt

# MOTIVEN
from candidate_state_evaluator import CandidateStateEvaluator
from episode import Episode
from episodic_buffer import EpisodicBuffer
from goal_manager import GoalManager
from goal_manager_mdp import GoalManagerMDP
from simulator import Sim
from surs_manager import SURsManager
from traces_buffer import TracesBuffer
from traces_memory import TracesMemory

import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.managers import NamespaceProxy, BaseManager
import types

import time

from info_teory_disc import InfoDiscrete


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
    

class MOTIVEN(object):
    def __init__(self):
        
        self.db = False
        
        self.time_main_starts = None
        self.time_main_ends = None
        self.time_data_starts = None
        self.time_data_ends = None
        
        self.extrinsic_goal = "ball_1"
        self.sim_data = None 
        self.action_data = None 
        self.sensors_data = None 
        self.goal_data = None
        self.target_reached = None 
        self.reward = None
        
        self.iter_count = None 
        self.action = None
        self.c_epoch = None
        
        self.agent_id = None
        
        # np.random.set_state(1)
        np.random.seed(1)
        # self.map_elites_eval_epoch = 1000
        
        # Object initialization
        self.memory_vf = TracesBuffer(max_size=100)
        self.traces_memory_vf = TracesMemory()

        self.simulator = Sim()
        self.traces_buffer = TracesBuffer(max_size=15)
        self.intrinsic_memory = EpisodicBuffer(max_size=200)

        self.episode = Episode()
        self.surs_manager = SURsManager(n_sensors=len(self.simulator.get_sensorization()))
        self.candidate_state_evaluator = CandidateStateEvaluator()

        # Variables initialization
        self.iterations = 0
        self.it_trial = 0  # Number of iterations before obtaining reward or ending the trial
        self.it_blind = 0  # Number of iterations the Intrinsic blind motivation is active
        self.n_trials = 0  # Number of trial of the experiment
        self.n_epochs = 0  # Number of epoch of the experiment

        # IMPORTANT: DATA TO CHANGE IN BETWEEN EXPERIMENTS
        self.max_iterations = 20000  # 10000  # Iterations to finish the experiment
        
        # TODO: CHANGE MAX EPOCH
        self.max_epochs = None # 1500  # 700  # Number of epochs to finish the experiment
        self.max_trials = 12  # Number of trials per epoch
        self.max_it_trial = 70  # 100#250  # Max number of iterations per trial (to avoid getting stuck)
        self.epochs_save_data = 25  # 25#20  # epoch interval after which to save the models to evaluate their performance
        self.filename = 'ums_performance_5_11'
        self.folder = 'D:/EVO_GRAIL/temp_1_v2/' #'performance_data/finales/'

        self.simulator.visualize = False
        
        # self.reward_extrinsic = np.zeros(self.max_epochs+1)
        #######

        self.active_mot = 'Int'  # Variable to control the active motivation: Intrinsic ('Int') or Extrinsic ('Ext')

        self.active_corr = 0  # Variable to control the active correlation. It contains its index
        self.corr_sensor = 0  # 1 - Sensor 1, 2 - Sensor 2, ... n- sensor n, 0 - no hay correlacion
        self.corr_type = ''  # 'pos' - Positive correlation, 'neg' - Negative correlation, '' - no correlation

        self.iter_min = 0  # Minimum number of iterations to consider possible an antitrace

        self.use_motiv_manager = True

        ### GOAL MANAGER ###
        self.use_goal_selector = True
        self.use_mdp = False  # True
        if self.use_goal_selector:
            if self.use_mdp:
                self.goal_manager_mdp = GoalManagerMDP()
                goals_list = ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'ball_6'] #, 'ball_7', 'ball_8']
                for goal in goals_list:
                    self.goal_manager_mdp.add_goal(goal)
            else:
                self.goal_manager = GoalManager()
                goals_list = ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'ball_6'] #, 'ball_7', 'ball_8']
                for goal in goals_list:
                    self.goal_manager.add_goal(goal)
        # MDP
        self.scenario_mdp = self.use_mdp  # True  # Reinicio parcial del escenario para aprender las cadenas de goals
        self.state_t = self.simulator.get_goals_state()
        self.state_t1 = self.simulator.get_goals_state()
        self.final_state = tuple([True for i in range(len(goals_list))])

        # Graphs
        self.graph1 = []
        self.graph2 = []

        self.end_trial = False

        # Evaluate performance
        self.evaluating_performance = False
        self.evaluating_performance_b = False
        if self.use_mdp:
            self.max_epochs_performance = len(self.goal_manager_mdp.goals_list) * 10
        else:
            self.max_epochs_performance = len(self.goal_manager.goals_list) * 10
        self.epochs_performance = 0
        self.graph_performance = {}  # Save performance in a dict, keys are the evaluation epochs
        self.graph_performance_b = {}
    
        
    def run(self):
        # Save/load seed
        ans = input("Do you want to load an existing seed? (yes/no) ")
        if ans == 'yes':
            name = input("Enter existing seed name: ")
            self.load_seed(seed_name=name)
        else:
            name = input("Enter new seed name: ")
            self.save_seed(seed_name=name)
        self.main()

    def load_seed(self, seed_name):
        """Import seed"""
        f = open(seed_name + '.pckl', 'rb')
        seed = pickle.load(f)  # , encoding='latin1')
        f.close()
        np.random.set_state(seed)

    def save_seed(self, seed_name):
        """Save seed"""
        seed = np.random.get_state()
        f = open(seed_name + '.pckl', 'wb')
        pickle.dump(seed, f)
        f.close()
        
    
    def test_agent(self, goal, n_test=100):
        self.simulator.set_active_goal(goal)
        self.goal_manager.current_goal = goal
        reward = 0
        for i in range(n_test):
            # test the guy perform an action
            if self.simulator.ball_1_active and goal == "ball_1":
                reward += 1
            elif self.simulator.ball_2_active and goal == "ball_2":
                reward += 1
        
        return reward / n_test
    
    
    # TODO: Alejandro help here!!
    # def get_data(self, n_test=100):
    def get_data(self):    
        
        # self.db = True
        # self.time_data_starts = time.time()
        self.surs_goal_list = []
        for i in range(len(self.surs_manager.surs)):
            self.surs_goal_list.append(self.surs_manager.surs[i].goal)
        self.surs_goal_list = np.array(self.surs_goal_list)
        
        
        n_test = 100
        
        # sim_data = [] # sensorization used to get sensory state 
        action_data = [] #np.zeros(n_test, dtype=object) # save the action at each time step of the trial
        sensors_data = [] #np.zeros(n_test, dtype=object) # get the distance from the robot's effector and the target (if this is too much work do not worry, I think it is redundant)
        goal_data_string = []
        target_reached = np.zeros(n_test)
        self.reward_test = 0
        
        
        for n_t in range(n_test):
            c_goal = self.goal_manager.goal_selector('c0')[1]
            # if np.any(self.surs_goal_list==self.goal_manager.goal_selector('c0')[1]):
            while(~np.any(self.surs_goal_list==c_goal)):
                  print("unknown goal")
                  c_goal = self.goal_manager.goal_selector('c0')[1]
            
            goal_data_string.append(c_goal)
            # I do not use this line. I do not want parameters to be updated
            # self.goal_manager.current_goal = active_goal
            self.simulator.set_active_goal(c_goal)  
            
            
            self.simulator.restart_scenario()
            # In the case goals have precondition, you should set the reconditions as active. Something like this:
            # if active_goal == 'ball_1':
            #     self.simulator.ball_2_active = False
            #     # self.simulator.ball_2.set_alpha(1.0)
            # elif active_goal == 'ball_2':
            #     self.simulator.ball_1_active = False
            #     # self.simulator.ball_1.set_alpha(1.0)

            # sim_data_trial = [] # sensorization used to get sensory state 
            action_data_trial = [] # save the action at each time step of the trial
            sensors_data_trial = [] # get the distance from the robot's effector and the target (if this is too much work do not worry, I think it is redundant)
            
            
            
            for _ in range(self.max_it_trial):
                
                # perform a sequence action and get simulator data
                action_test = self.method_to_get_action_without_learning() # <- I think this is the only thing I need
                                
                action_data_trial.append(action_test)
                self.simulator.apply_action(action_test)
                # sim_data_trial.append(self.simulator.get_scenario_data())
                sensors_data_trial.append(self.simulator.get_sensorization())
                if self.simulator.get_reward():  #the current target ball has been reached
                    target_reached[n_t] = 1   
                    if self.extrinsic_goal == goal_data_string[-1]:
                        self.reward_test += 1
                    break

            # sim_data.append(sim_data_trial)
            # self.sensors_data[n_t] = np.array(sensors_data_trial)
            sensors_data.append(np.array(sensors_data_trial))
            # self.action_data[n_t] = np.array(self.action_data_trial)
            action_data.append(np.array(action_data_trial))
            
        
        self.reward_test /= n_test    
        
        sensors_data = np.vstack(sensors_data)
        action_data = np.hstack(action_data)
        action_data = InfoDiscrete.range_scaler(action_data, 
                                                min_bound=-90.0, 
                                                max_bound=90.0)
        goal_data = np.zeros(len(goal_data_string))
        for i in range(1,7):
            g = "ball_" + str(i)
            goal_data[np.where(np.array(goal_data_string)==g)] = i
            
        self.entropy_goal = InfoDiscrete.entropy(data=goal_data, 
                                                 n_bins=6, 
                                                 states_range=[1,6],
                                                 normalize=False)
        
        self.entropy_sensors = InfoDiscrete.entropy(data=sensors_data.flatten(), 
                                                    n_bins=30,
                                                    states_range=[0.,1],
                                                    normalize=False)
        
        self.entropy_action = InfoDiscrete.entropy(data=action_data, 
                                                   n_bins=30,
                                                   states_range=[0.,1],
                                                   normalize=False)
        
        action_expanded = repmat(action_data, n=1, m=6).T
        entropy_sensors_action = InfoDiscrete.joint_entropy(data_x=action_expanded.flatten(), 
                                                            data_y=sensors_data.flatten(), 
                                                            n_bins=[30, 30], 
                                                            states_range=[[0.,1.],[0.,1.]])
        # print(entropy_sensors_action)
        self.I_sensors_action = InfoDiscrete.mutual_information(E_x=self.entropy_action,
                                                                E_y=self.entropy_sensors,
                                                                E_xy=entropy_sensors_action,
                                                                normalization="IQR")
        
        self.entropy_goal /= np.log(6)
        self.entropy_action /= np.log(30)
        self.entropy_sensors /= np.log(30)
        
        
    def method_to_get_action_without_learning(self):
        sim_data = self.simulator.get_scenario_data()
        self.candidate_state_evaluator.intrinsic_exploration_type = 'Brownian'
        self.use_motiv_manager = True
        self.motivation_manager()  # Just to know which Utility Model to use
        action = self.candidate_state_evaluator.get_action(
            exploration_type=self.active_mot,
            sim_data=sim_data,
            sens_t1=tuple(self.simulator.get_sensorization()),
            corr_sensor=self.corr_sensor,
            corr_type=self.corr_type,
            intrinsic_memory=self.intrinsic_memory.get_contents()
        )
        return action
    
    def init_main(self):
        self.iterations = 0
        self.action = self.candidate_state_evaluator.action_chooser.get_candidate_actions(1)[0]
        self.simulator.choose_active_context(self.iterations)
        self.goal_selector(update_competence=False, attempt=0.0, state_t=self.state_t, state_t1=self.state_t1)
        
        # self.c_epoch = self.n_epochs
        
        self.iter_count = 0

    def main(self):
        
        # self.time_main_starts = time.time()
        
        self.extrinsic_goal = "ball_1"
        # self.simulator.restart_scenario()
        
        while self.n_epochs <= self.max_epochs:
           
            
            # self.iter_count1 += 1
            if self.n_epochs == 600:
                self.extrinsic_goal = "ball_2"
            
            # if self.c_epoch != self.n_epochs:
            #     self.reward_extrinsic[self.c_epoch] /= (self.iter_count - 1)
            #     # print(self.reward_extrinsic[c_epoch])
            #     self.iter_count = 0
              
            # self.c_epoch = self.n_epochs
            
            
                        
            if not self.use_goal_selector:
                self.simulator.choose_active_goal(self.iterations)
            # Perception in t (distances, action and motivation)
            self.episode.clean_episode()
            self.episode.set_sensorial_state_t(self.simulator.get_sensorization())
            self.episode.set_action(self.action)
            self.simulator.apply_action(self.action)
            # Perception in t+1 (distances and reward)
            self.episode.set_sensorial_state_t1(self.simulator.get_sensorization())
            self.episode.set_reward(self.simulator.get_reward())
            # if self.extrinsic_goal == "ball_1" and self.simulator.ball_1_active:
            #     self.reward_extrinsic[self.n_epochs] += 1
            # elif self.extrinsic_goal == "ball_2" and self.simulator.ball_2_active:
            #     self.reward_extrinsic[self.n_epochs] += 1
            
            
            
            # MEMORY MANAGER
            self.memory_manager()
            
            
            # Check if the end of trial or epoch has been reached
            if self.evaluating_performance or self.evaluating_performance_b:
                self.check_scenario_restart_performance()
            else:
                self.check_scenario_restart()
                if self.n_epochs % self.epochs_save_data == 0:  # and self.n_epochs > 0:
                    self.evaluating_performance = True
                    self.graph_performance[self.n_epochs] = []
                    self.graph_performance_b[self.n_epochs] = []
                    self.candidate_state_evaluator.intrinsic_exploration_type = 'Brownian'
                    self.restart_scenario_manually(partially=False)
                    # print("B")
                else:
                    self.evaluating_performance = False
                    self.candidate_state_evaluator.intrinsic_exploration_type = 'Novelty'
                    # print("N")
                    
            
            
            # MOTIVATION MANAGER
            self.motivation_manager()
            # CANDIDATE STATE EVALUATOR and ACTION CHOOSER
            # Generate new action
            self.sim_data = self.simulator.get_scenario_data()
            self.action = self.candidate_state_evaluator.get_action(
                exploration_type=self.active_mot,
                sim_data=self.sim_data,
                sens_t1=tuple(self.simulator.get_sensorization()),
                corr_sensor=self.corr_sensor,
                corr_type=self.corr_type,
                intrinsic_memory=self.intrinsic_memory.get_contents()
            )
            
        self.get_data()    
      
            # self.n_epochs += 1      
        

    def check_scenario_restart(self):
        end_epoch = False
        end_trial = False
        self.it_trial += 1
        if self.it_trial >= self.max_it_trial or self.simulator.get_goals_state() != self.state_t or self.end_trial:  # Antitraza
            end_trial = True
            self.n_trials += 1
            self.it_trial = 0
            self.end_trial = False
        if self.n_trials >= self.max_trials or (self.simulator.get_reward() and not self.scenario_mdp) or (
                all(x is True for x in self.simulator.get_goals_state())):
            end_epoch = True
            self.n_epochs += 1
            self.n_trials = 0
            self.it_trial = 0
        if end_epoch or end_trial:
            competence = True
            if self.simulator.get_goals_state() != self.state_t:
                if self.active_mot == 'Int':
                    attempt = 0.5 * self.episode.get_reward()  # 0.5*competence
                    # print("check_scenario_restart Int: " + str(attempt))
                else:
                    attempt = 1.0 * self.episode.get_reward()  # 1.0*competence
                    # print("check_scenario_restart not Int: " + str(attempt))
            else:
                attempt = 0.0
            self.restart_scenario(update_competence=competence, attempt=attempt, partially=not end_epoch)
        # OTHERS
        # self.debug_print()
        self.iter_min += 1
        self.iterations += 1

    def restart_scenario(self, update_competence, attempt, partially):
        # print("restart_scenario: " + str(attempt))
        self.state_t1 = self.simulator.get_goals_state()  # End of the trial
        self.graph1.append((self.state_t1, self.n_epochs, self.n_trials))
        if partially:
            # I must save the reward before restarting the scenario
            self.simulator.restart_scenario_partially()
            if self.scenario_mdp:
                self.goal_selector(update_competence=update_competence, attempt=attempt, state_t=self.state_t,
                                   state_t1=self.state_t1)
        else:
            self.simulator.restart_scenario()  # Restart scenario
            self.goal_selector(update_competence=update_competence, attempt=attempt, state_t=self.state_t,
                               state_t1=self.state_t1)
        self.state_t = self.simulator.get_goals_state()  # Start of the trial
        # ----------------
        # Check if a new correlation is needed or established
        self.surs_manager.new_sur(self.simulator.active_goal, self.simulator.active_context)
        # ----------------
        self.active_corr = self.surs_manager.get_active_sur(
            self.simulator.get_sensorization(), self.simulator.active_goal, self.simulator.active_context)
        self.it_blind = 0
        self.reinitialize_memories()
        self.memory_vf.remove_all()

        self.use_motiv_manager = True

    def check_scenario_restart_performance(self):
        # print("---EVALUATING PERFORMANCE---")
        end_epoch = False
        end_trial = False
        self.it_trial += 1
        if self.it_trial >= self.max_it_trial or self.simulator.get_goals_state() != self.state_t or self.end_trial:  # Antitraza
            end_trial = True
            self.n_trials += 1
            self.it_trial = 0
            self.end_trial = False
        if (self.n_trials >= 1 and self.evaluating_performance) or (
                self.n_trials >= self.max_trials and self.evaluating_performance_b) or self.simulator.get_reward():
            end_epoch = True
            self.epochs_performance += 1
            if self.use_mdp:
                if not self.evaluating_performance_b:
                    self.graph_performance[self.n_epochs].append(
                        (self.goal_manager_mdp.current_goal, self.simulator.get_reward(), self.n_trials))
                else:
                    self.graph_performance_b[self.n_epochs].append(
                        (self.goal_manager_mdp.current_goal, self.simulator.get_reward(), self.n_trials))
            else:
                if not self.evaluating_performance_b:
                    self.graph_performance[self.n_epochs].append(
                        (self.goal_manager.current_goal, self.simulator.get_reward(), self.n_trials))
                else:
                    self.graph_performance_b[self.n_epochs].append(
                        (self.goal_manager.current_goal, self.simulator.get_reward(), self.n_trials))
            self.n_trials = 0
            self.it_trial = 0
            if self.epochs_performance >= self.max_epochs_performance:
                if self.evaluating_performance_b:
                    self.evaluating_performance_b = False
                    self.n_epochs += 1
                else:
                    self.evaluating_performance = False
                    self.evaluating_performance_b = True
                self.epochs_performance = 0
        if end_epoch or end_trial:
            if self.evaluating_performance or self.evaluating_performance_b:
                self.restart_scenario_manually(partially=not end_epoch)
            else:
                self.restart_scenario(update_competence=False, attempt=0.0, partially=False)
        # OTHERS
        # self.debug_print()

    def restart_scenario_manually(self, partially):
        self.state_t1 = self.simulator.get_goals_state()  # End of the trial
        if partially:
            self.simulator.restart_scenario_partially()
        else:
            self.simulator.restart_scenario()  # Restart scenario
        if self.use_mdp:
            active_goal = self.goal_manager_mdp.goals_list[
                self.epochs_performance // (self.max_epochs_performance / len(self.goal_manager_mdp.goals_list))]
            self.goal_manager_mdp.current_goal = active_goal
            self.simulator.set_active_goal(self.goal_manager_mdp.current_goal)
        else:
            active_goal = self.goal_manager.goals_list[
                int(self.epochs_performance // (self.max_epochs_performance / len(self.goal_manager.goals_list)))]
            self.goal_manager.current_goal = active_goal
            self.simulator.set_active_goal(self.goal_manager.current_goal)
        if not self.evaluating_performance_b:
            if active_goal == 'ball_1':
                self.simulator.ball_2_active = False
                # self.simulator.ball_2.set_alpha(1.0)
            elif active_goal == 'ball_2':
                self.simulator.ball_1_active = False
                # self.simulator.ball_1.set_alpha(1.0)
            elif active_goal == 'ball_3':
                self.simulator.ball_1_active = False
                # self.simulator.ball_1.set_alpha(1.0)
                # self.simulator.ball_2_active = True
                # self.simulator.ball_2.set_alpha(1.0)
            elif active_goal == 'ball_4':
                self.simulator.ball_1_active = True
                self.simulator.ball_1.set_alpha(1.0)
            elif active_goal == 'ball_5':
                self.simulator.ball_2_active = True
                self.simulator.ball_2.set_alpha(1.0)
                self.simulator.ball_3_active = True
                self.simulator.ball_3.set_alpha(1.0)
            elif active_goal == 'ball_6':
                self.simulator.ball_1_active = True
                self.simulator.ball_1.set_alpha(1.0)
                self.simulator.ball_2_active = True
                self.simulator.ball_2.set_alpha(1.0)
                self.simulator.ball_3_active = True
                self.simulator.ball_3.set_alpha(1.0)
                self.simulator.ball_5_active = True
                self.simulator.ball_5.set_alpha(1.0)
        self.state_t = self.simulator.get_goals_state()  # Start of the trial
        # ----------------
        # Check if a new correlation is needed or established
        self.surs_manager.new_sur(self.simulator.active_goal, self.simulator.active_context)
        # ----------------
        self.active_corr = self.surs_manager.get_active_sur(
            self.simulator.get_sensorization(), self.simulator.active_goal, self.simulator.active_context)
        self.reinitialize_memories()
        self.memory_vf.remove_all()
        self.use_motiv_manager = True

    def debug_print(self):
        if not self.evaluating_performance and not self.evaluating_performance_b:
            print("Number of epochs: ", self.n_epochs)

    def reinitialize_memories(self):
        self.traces_buffer.remove_all()  # Reinitialize traces buffer
        self.iter_min = 0
        self.intrinsic_memory.remove_all()  # Reinitialize intrinsic memory
        self.intrinsic_memory.add_episode(self.episode.get_sensorial_state_t1())

    def goal_selector(self, update_competence, attempt, state_t, state_t1):
        if self.use_goal_selector:
            if self.use_mdp:  # With MDP
                if update_competence:
                    self.goal_manager_mdp.update_competence(
                        goal_id=self.goal_manager_mdp.current_goal,
                        attempt_result=attempt,
                        goals_state_t=state_t,
                        goal_state_t1=state_t1)
                    self.simulator.choose_active_context(self.iterations)
                prob, goal, q_goals = self.goal_manager_mdp.goal_selector(state=self.simulator.get_goals_state())
                self.simulator.set_active_goal(self.goal_manager_mdp.current_goal)
            else:  # Without MDP
                if update_competence:
                    self.goal_manager.update_competence(self.goal_manager.current_goal, attempt_result=attempt,
                                                        context=self.simulator.active_context)
                    self.simulator.choose_active_context(self.iterations)
                prob, goal, q_goals = self.goal_manager.goal_selector(context=self.simulator.active_context)
                self.simulator.set_active_goal(self.goal_manager.current_goal)
            self.graph2.append((prob, goal, q_goals))

    def motivation_manager(self):
        if self.use_motiv_manager:
            self.corr_sensor, self.corr_type = self.surs_manager.get_active_correlation(
                p=tuple(self.simulator.get_sensorization()),
                active_corr=self.active_corr,
                active_goal=self.simulator.active_goal,
                context=self.simulator.active_context,
                goals_state=self.state_t
            )
            if self.corr_sensor == 0:
                self.active_mot = 'Int'
            else:
                if self.active_mot == 'Int':
                    self.iter_min = 0
                self.active_mot = 'Ext'

    def memory_manager(self):
        # Save episode in the pertinent memories
        self.traces_buffer.add_episode(self.episode.get_episode())
        self.memory_vf.add_episode(self.episode.get_episode())
        self.intrinsic_memory.add_episode(self.episode.get_sensorial_state_t1())
        # Check if a new correlation is needed or established
        self.surs_manager.new_sur(self.simulator.active_goal, self.simulator.active_context)
        if not self.surs_manager.surs[self.active_corr].i_reward_assigned:
            self.surs_manager.assign_reward_assigner(self.active_corr, self.episode.get_sensorial_state_t1(),
                                                     self.simulator.active_goal, self.simulator.active_context)
        # Check if a new skill is needed or established
        # Memory Manager (Traces, weak traces and antitraces)
        if self.active_mot == 'Int':
            self.it_blind += 1
            self.use_motiv_manager = True
            # If there is a reward, realise reward assignment and save trace in Traces Memory
            if self.episode.get_reward():
                ###
                if not self.surs_manager.surs[self.active_corr].i_reward_assigned:
                    self.surs_manager.assign_reward_assigner(self.active_corr,
                                                             self.episode.get_sensorial_state_t1(),
                                                             self.simulator.active_goal, self.simulator.active_context,
                                                             1)
                ###
                self.surs_manager.surs[self.active_corr].correlation_evaluator(
                    trace=self.traces_buffer.get_trace(),
                    goals_state=self.state_t
                )
                self.traces_memory_vf.add_traces(self.memory_vf.get_trace_reward())
            elif self.surs_manager.get_reward(self.active_corr, self.simulator.get_reward(),
                                              tuple(self.episode.get_sensorial_state_t1()),
                                              self.simulator.active_goal, self.simulator.active_context):
                self.surs_manager.surs[self.active_corr].correlation_evaluator(
                    trace=self.traces_buffer.get_trace(),
                    goals_state=self.state_t,
                )
                # The active correlation is now the correlation that has provided the reward
                self.active_corr = self.surs_manager.surs[self.active_corr].i_reward
                self.reinitialize_memories()
        elif self.active_mot == 'Ext':
            self.use_motiv_manager = False
            if self.episode.get_reward():
                # Save as trace in traces_memory of the correlated sensor
                self.surs_manager.surs[self.active_corr].add_trace(
                    trace=self.traces_buffer.get_trace(),
                    sensor=self.corr_sensor,
                    corr_type=self.corr_type,
                    goals_state=self.state_t
                )
                self.traces_memory_vf.add_traces(self.memory_vf.get_trace_reward())
                self.use_motiv_manager = True
            elif self.surs_manager.get_reward(self.active_corr, self.simulator.get_reward(),
                                              tuple(self.episode.get_sensorial_state_t1()),
                                              self.simulator.active_goal, self.simulator.active_context):
                # Save as trace in traces_memory of the correlated sensor
                self.surs_manager.surs[self.active_corr].add_trace(self.traces_buffer.get_trace(),
                                                                   self.corr_sensor, self.corr_type)
                # The active correlation is now the correlation that has provided the reward
                self.active_corr = self.surs_manager.surs[self.active_corr].i_reward
                self.reinitialize_memories()
                self.use_motiv_manager = True
            else:
                # Check if the the active correlation is still active
                if self.iter_min > 2:
                    sens_t = self.traces_buffer.get_trace()[-2][self.corr_sensor - 1]
                    sens_t1 = self.traces_buffer.get_trace()[-1][self.corr_sensor - 1]
                    dif = sens_t1 - sens_t
                    if (self.corr_type == 'pos' and dif <= 0) or (self.corr_type == 'neg' and dif >= 0):
                        self.surs_manager.surs[self.active_corr].add_antitrace(
                            trace=self.traces_buffer.get_trace(),
                            sensor=self.corr_sensor,
                            corr_type=self.corr_type,
                            goals_state=self.state_t
                        )
                        self.end_trial = True
                        self.use_motiv_manager = True

    @staticmethod
    def calc_sma(data, sma_period):
        j = next(i for i, x in enumerate(data) if x is not None)
        our_range = range(len(data))[j + sma_period - 1:]
        empty_list = [None] * (j + sma_period - 1)
        sub_result = [np.mean(data[i - sma_period + 1: i + 1]) for i in our_range]
        return list(empty_list + sub_result)

    def save_data(self, filename):
        f = open(filename + '.pckl', 'wb')
        pickle.dump(self.surs_manager, f)
        pickle.dump(self.simulator, f)
        pickle.dump(self.candidate_state_evaluator, f)
        pickle.dump(self.max_it_trial, f)
        f.close()

    def load_data(self, filename):
        f = open(filename + '.pckl', 'rb')
        self.surs_manager = pickle.load(f)
        self.simulator = pickle.load(f)
        self.candidate_state_evaluator = pickle.load(f)
        self.max_it_trial = pickle.load(f)
        f.close()

    def plot_goal_selections(self):
        # Graph 1: Goal states
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.graph1)):
            # Goal States
            state = self.graph1[i][0]
            for j in range(len(state)):
                if state[j]:
                    ax.plot(i, j + 1, 'ro', color='green')  # Ball ON
                else:
                    ax.plot(i, j + 1, 'ro', color='red')  # Ball OFF
            if self.graph1[i][1] > self.graph1[i - 1][1]:  # new epoch
                ax.axvline(x=i, color='grey', linestyle='--')

        ax.set_xlabel('Trials', size=12.0)
        ax.set_ylabel('Goals', size=12.0)
        plt.yticks([1, 2, 3, 4, 5, 6], ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Ball 6'],
                   rotation='horizontal')
        ax.set_title('Goal states (end of trial)', size=12.0)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='ON', markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='OFF', markerfacecolor='red', markersize=10),
        ]
        ax.legend(handles=legend_elements)
        plt.show()

        # Graph 2: Goal selections epochs
        p1 = []
        p2 = []
        p3 = []
        p4 = []
        p5 = []
        p6 = []
        q1 = []
        q2 = []
        q3 = []
        q4 = []
        q5 = []
        q6 = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.graph2)):
            # Reorganize values to plot thgrem
            # Probabilities after softmax
            p1.append(self.graph2[i][0][0])
            p2.append(self.graph2[i][0][1])
            p3.append(self.graph2[i][0][2])
            p4.append(self.graph2[i][0][3])
            p5.append(self.graph2[i][0][4])
            p6.append(self.graph2[i][0][5])
            # Goal q values
            q1.append(self.graph2[i][2][0])
            q2.append(self.graph2[i][2][1])
            q3.append(self.graph2[i][2][2])
            q4.append(self.graph2[i][2][3])
            q5.append(self.graph2[i][2][4])
            q6.append(self.graph2[i][2][5])
        epochs = list(range(len(self.graph2)))
        ax.plot(epochs, p1, marker='.', color='red', linewidth=0.5, label='Goal1')
        ax.plot(epochs, p2, marker='.', color='green', linewidth=0.5, label='Goal2')
        ax.plot(epochs, p3, marker='.', color='blue', linewidth=0.5, label='Goal3')
        ax.plot(epochs, p4, marker='.', color='orange', linewidth=0.5, label='Goal4')
        ax.plot(epochs, p5, marker='.', color='purple', linewidth=0.5, label='Goal5')
        ax.plot(epochs, p6, marker='.', color='cyan', linewidth=0.5, label='Goal6')
        plt.legend()
        ax.grid()
        ax.set_xlabel('Epochs', size=12.0)
        ax.set_ylabel('Probabilities', size=12.0)
        ax.set_title('Goal selection probabilities', size=12.0)
        plt.show()
        # plt.savefig('goal_prob_temp_0_03.png', dpi=200)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(epochs, q1, marker='.', color='red', linewidth=0.5, label='Goal1')
        ax.plot(epochs, q2, marker='.', color='green', linewidth=0.5, label='Goal2')
        ax.plot(epochs, q3, marker='.', color='blue', linewidth=0.5, label='Goal3')
        ax.plot(epochs, q4, marker='.', color='orange', linewidth=0.5, label='Goal4')
        ax.plot(epochs, q5, marker='.', color='purple', linewidth=0.5, label='Goal5')
        ax.plot(epochs, q6, marker='.', color='cyan', linewidth=0.5, label='Goal6')
        plt.legend()
        ax.grid()
        ax.set_xlabel('Epochs', size=12.0)
        ax.set_ylabel('Values', size=12.0)
        ax.set_title('Goal q values', size=12.0)
        plt.show()
        # plt.savefig('goal_q_values_temp_0_03.png', dpi=200)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cdict = {'ball_1': ['red', 1],
                 'ball_2': ['green', 2],
                 'ball_3': ['blue', 3],
                 'ball_4': ['orange', 4],
                 'ball_5': ['purple', 5],
                 'ball_6': ['cyan', 6]}
        for i in range(len(self.graph2)):
            ax.plot(i, cdict[self.graph2[i][1]][1], 'ro', color=cdict[self.graph2[i][1]][0])
        ax.grid()
        ax.set_xlabel('Epochs', size=12.0)
        ax.set_ylabel('Goals', size=12.0)
        ax.set_title('Goal selection epochs', size=12.0)
        plt.legend()
        plt.show()
        # plt.savefig('goal_selection_temp_0_03.png', dpi=200)

        # Graph 3: subgoal selections trials
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # for i in range(len(self.graph3)):
        #     ax.plot(i, cdict[self.graph3[i][0]][1], 'ro', color=cdict[self.graph3[i][0]][0])
        #     if self.graph1[i][1] > self.graph1[i - 1][1]:  # new epoch
        #         ax.axvline(x=i, color='grey', linestyle='--')
        # ax.grid()
        # ax.set_xlabel('Trials', size=12.0)
        # ax.set_ylabel('SubGoals', size=12.0)
        # ax.set_title('SubGoal selection trials', size=12.0)
        # plt.legend()
        # plt.show()

        # Graph 3: goal/subgoal selections trials
        fig = plt.figure()
        ax = fig.add_subplot(111)
        legend_elements = []
        for key in cdict.keys():
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=cdict[key][0], markersize=10))
        for i in range(len(self.graph3)):
            ax.plot(i, 1, 'ro', color=cdict[self.graph3[i][0]][0])  # Subgoal selection
            ax.plot(i, 2, 'ro', color=cdict[self.graph3[i][1]][0])  # Goal slection
            if self.graph1[i][1] > self.graph1[i - 1][1]:  # new epoch
                ax.axvline(x=i, color='grey', linestyle='--')
        ax.grid()
        ax.set_xlabel('Trials', size=12.0)
        ax.set_ylabel('Selections', size=12.0)
        plt.yticks([1, 2], ['Subgoal', 'Goal'], rotation='horizontal')
        ax.set_title('Goal and SubGoal selection', size=12.0)
        ax.legend(handles=legend_elements, loc='center left', ncol=6)
        plt.show()
        print("END FIGURES")

    def plot_performance(self):
        data_performance = {}
        data_performance['ball_1'] = []
        data_performance['ball_2'] = []
        data_performance['ball_3'] = []
        data_performance['ball_4'] = []
        data_performance['ball_5'] = []
        data_performance['ball_6'] = []
        data_it_goal = {}
        data_it_goal['ball_1'] = []
        data_it_goal['ball_2'] = []
        data_it_goal['ball_3'] = []
        data_it_goal['ball_4'] = []
        data_it_goal['ball_5'] = []
        data_it_goal['ball_6'] = []
        for key in sorted(self.graph_performance.keys()):
            epoch_data = self.graph_performance[key]
            data_performance['ball_1'].append(sum(list(zip(*epoch_data[0:10]))[1]))
            data_performance['ball_2'].append(sum(list(zip(*epoch_data[10:20]))[1]))
            data_performance['ball_3'].append(sum(list(zip(*epoch_data[20:30]))[1]))
            data_performance['ball_4'].append(sum(list(zip(*epoch_data[30:40]))[1]))
            data_performance['ball_5'].append(sum(list(zip(*epoch_data[40:50]))[1]))
            data_performance['ball_6'].append(sum(list(zip(*epoch_data[50:60]))[1]))
            data_it_goal['ball_1'].append(sum(list(zip(*epoch_data[0:10]))[2]))
            data_it_goal['ball_2'].append(sum(list(zip(*epoch_data[10:20]))[2]))
            data_it_goal['ball_3'].append(sum(list(zip(*epoch_data[20:30]))[2]))
            data_it_goal['ball_4'].append(sum(list(zip(*epoch_data[30:40]))[2]))
            data_it_goal['ball_5'].append(sum(list(zip(*epoch_data[40:50]))[2]))
            data_it_goal['ball_6'].append(sum(list(zip(*epoch_data[50:60]))[2]))

        # Plot performance: para ver si se aprenden los modelos
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data_performance['ball_1'], color='red', label='Goal 1')
        ax.plot(data_performance['ball_2'], color='green', label='Goal 2')
        ax.plot(data_performance['ball_3'], color='blue', label='Goal 3')
        ax.plot(data_performance['ball_4'], color='orange', label='Goal 4')
        ax.plot(data_performance['ball_5'], color='purple', label='Goal 5')
        ax.plot(data_performance['ball_6'], color='cyan', label='Goal 6')
        plt.legend()
        ax.grid()
        ax.set_title('Bandit + UMs')
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Performance', size=12)
        plt.xticks(list(range(len(data_performance['ball_1']))), sorted(self.graph_performance.keys()))
        plt.savefig(self.folder + self.filename + "_performance_a.png", dpi=200)

        # Plot iterations:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        media_ball_1 = [x / 10.0 for x in data_it_goal['ball_1']]
        ax.plot(media_ball_1, color='red', label='Goal 1')
        media_ball_2 = [x / 10.0 for x in data_it_goal['ball_2']]
        ax.plot(media_ball_2, color='green', label='Goal 2')
        media_ball_3 = [x / 10.0 for x in data_it_goal['ball_3']]
        ax.plot(media_ball_3, color='blue', label='Goal 3')
        media_ball_4 = [x / 10.0 for x in data_it_goal['ball_4']]
        ax.plot(media_ball_4, color='orange', label='Goal 4')
        media_ball_5 = [x / 10.0 for x in data_it_goal['ball_5']]
        ax.plot(media_ball_5, color='purple', label='Goal 5')
        media_ball_6 = [x / 10.0 for x in data_it_goal['ball_6']]
        ax.plot(media_ball_6, color='cyan', label='Goal 6')
        plt.legend()
        ax.grid()
        ax.set_title('Bandit + UMs')
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Average trials to reach goal', size=12)
        plt.xticks(list(range(len(media_ball_1))), sorted(self.graph_performance.keys()))

        data_performance_b = {}
        data_performance_b['ball_1'] = []
        data_performance_b['ball_2'] = []
        data_performance_b['ball_3'] = []
        data_performance_b['ball_4'] = []
        data_performance_b['ball_5'] = []
        data_performance_b['ball_6'] = []
        data_it_goal_b = {}
        data_it_goal_b['ball_1'] = []
        data_it_goal_b['ball_2'] = []
        data_it_goal_b['ball_3'] = []
        data_it_goal_b['ball_4'] = []
        data_it_goal_b['ball_5'] = []
        data_it_goal_b['ball_6'] = []
        for key in sorted(self.graph_performance_b.keys()):
            epoch_data = self.graph_performance_b[key]
            data_performance_b['ball_1'].append(sum(list(zip(*epoch_data[0:10]))[1]))
            data_performance_b['ball_2'].append(sum(list(zip(*epoch_data[10:20]))[1]))
            data_performance_b['ball_3'].append(sum(list(zip(*epoch_data[20:30]))[1]))
            data_performance_b['ball_4'].append(sum(list(zip(*epoch_data[30:40]))[1]))
            data_performance_b['ball_5'].append(sum(list(zip(*epoch_data[40:50]))[1]))
            data_performance_b['ball_6'].append(sum(list(zip(*epoch_data[50:60]))[1]))
            data_it_goal_b['ball_1'].append(sum(list(zip(*epoch_data[0:10]))[2]))
            data_it_goal_b['ball_2'].append(sum(list(zip(*epoch_data[10:20]))[2]))
            data_it_goal_b['ball_3'].append(sum(list(zip(*epoch_data[20:30]))[2]))
            data_it_goal_b['ball_4'].append(sum(list(zip(*epoch_data[30:40]))[2]))
            data_it_goal_b['ball_5'].append(sum(list(zip(*epoch_data[40:50]))[2]))
            data_it_goal_b['ball_6'].append(sum(list(zip(*epoch_data[50:60]))[2]))

        # Plot performance: para ver si se aprenden los modelos
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data_performance_b['ball_1'], color='red', label='Goal 1')
        ax.plot(data_performance_b['ball_2'], color='green', label='Goal 2')
        ax.plot(data_performance_b['ball_3'], color='blue', label='Goal 3')
        ax.plot(data_performance_b['ball_4'], color='orange', label='Goal 4')
        ax.plot(data_performance_b['ball_5'], color='purple', label='Goal 5')
        ax.plot(data_performance_b['ball_6'], color='cyan', label='Goal 6')
        plt.legend()
        ax.grid()
        ax.set_title('Bandit + UMs')
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Performance', size=12)
        plt.xticks(list(range(len(data_performance_b['ball_1']))), sorted(self.graph_performance_b.keys()))
        plt.savefig(self.folder + self.filename + "_performance_b.png", dpi=200)

        # Plot iterations:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        media_ball_1_b = [x / 10.0 for x in data_it_goal_b['ball_1']]
        ax.plot(media_ball_1_b, color='red', label='Goal 1')
        media_ball_2_b = [x / 10.0 for x in data_it_goal_b['ball_2']]
        ax.plot(media_ball_2_b, color='green', label='Goal 2')
        media_ball_3_b = [x / 10.0 for x in data_it_goal_b['ball_3']]
        ax.plot(media_ball_3_b, color='blue', label='Goal 3')
        media_ball_4_b = [x / 10.0 for x in data_it_goal_b['ball_4']]
        ax.plot(media_ball_4_b, color='orange', label='Goal 4')
        media_ball_5_b = [x / 10.0 for x in data_it_goal_b['ball_5']]
        ax.plot(media_ball_5_b, color='purple', label='Goal 5')
        media_ball_6_b = [x / 10.0 for x in data_it_goal_b['ball_6']]
        ax.plot(media_ball_6_b, color='cyan', label='Goal 6')
        plt.legend()
        ax.grid()
        ax.set_title('Bandit + UMs')
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Average trials to reach goal', size=12)
        plt.xticks(list(range(len(media_ball_1_b))), sorted(self.graph_performance_b.keys()))
        plt.savefig(self.folder + "UMs_trials_goal_4.png", dpi=200)
        plt.savefig(self.folder + self.filename + "_trials_goal_b.png", dpi=200)

    # def save_performance_data(self):
    #     f = open(self.folder + self.filename + '.pckl', 'wb')
    #     pickle.dump(self.graph_performance, f)
    #     pickle.dump(self.graph_performance_b, f)
    #     f.close()
    
    
    def set_softmax_temperature(self, temp):
        self.goal_manager.set_softmax_temp(temp)
        
    def get_softmax_temperature(self):
        return self.goal_manager.get_softmax_temp()
    
    
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


def main():
    instance= MOTIVEN.create()
    # instance = MOTIVEN()
    print("done")
    instance.set_softmax_temperature(0.01)
    # instance.run()
    instance.init_main()
    for i in range(1,5):
        instance.max_epochs = i
    instance.main()
    
    return instance


if __name__ == '__main__':
    exp = main()



# TODO: some notes
# self.goal_manager.goal_selector('c0') #we can generate dataset for entropy
# attempt
# TODO: modify def reinitialize_memories(self): to save intrinsic memories