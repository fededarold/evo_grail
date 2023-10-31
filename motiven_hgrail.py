import logging
import math
import pickle

import numpy as np
from matplotlib import pyplot as plt

# MOTIVEN
from candidate_state_evaluator import CandidateStateEvaluator
from episode import Episode
from episodic_buffer import EpisodicBuffer
from goal_manager import GoalManager
from simulator import Sim
from subgoal_manager_mdp import SubGoalManagerMDP
from surs_manager import SURsManager
from traces_buffer import TracesBuffer
from traces_memory import TracesMemory


class MOTIVEN_HGRAIL(object):
    def __init__(self):

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

        self.max_epochs = 3000  # 2000#1000  # Number of epochs to finish the experiment
        self.max_trials = 8  # 8  # Number of trials per epoch
        self.max_it_trial = 70  # 100#250  # Max number of iterations per trial (to avoid getting stuck)
        self.epochs_save_data = 50  # epoch interval after which to save the models to evaluate their performance
        self.filename = 'h_grail_icdl_7_6'
        self.folder = 'performance_data/finales/'

        self.simulator.visualize = False
        #######

        self.active_mot = 'Int'  # Variable to control the active motivation: Intrinsic ('Int') or Extrinsic ('Ext')

        self.active_corr = 0  # Variable to control the active correlation. It contains its index
        self.corr_sensor = 0  # 1 - Sensor 1, 2 - Sensor 2, ... n- sensor n, 0 - no hay correlacion
        self.corr_type = ''  # 'pos' - Positive correlation, 'neg' - Negative correlation, '' - no correlation

        self.iter_min = 0  # Minimum number of iterations to consider possible an antitrace

        self.use_motiv_manager = True

        ### GOAL MANAGER ###
        self.use_goal_selector = True
        self.use_mdp = True
        if self.use_goal_selector:
            self.goal_manager = GoalManager()
            self.sub_goal_manager = SubGoalManagerMDP(reinforcement_signal='reward')
            goals_list = ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'ball_6']  # , 'ball_7']# 'ball_8']
            for goal in goals_list:
                self.sub_goal_manager.add_goal(goal)
                self.goal_manager.add_goal(goal)
        # MDP
        self.scenario_mdp = self.use_mdp  # True  # Partial restart of the scenario in order to learn goal chains
        self.state_t = self.simulator.get_goals_state()
        self.state_t1 = self.simulator.get_goals_state()
        self.final_state = tuple([True for i in range(len(goals_list))])

        # Graphs
        self.graph1 = []
        self.graph2 = []
        self.graph3 = []
        self.graph4 = []  # Stores the temperature values for each goal

        self.end_trial = False

        # Evaluate performance
        self.evaluating_performance = False
        self.max_epochs_performance = len(self.goal_manager.goals_list) * 10
        self.epochs_performance = 0
        self.graph_performance = {}  # Save performance in a dict, keys are the evaluation epochs

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
        seed = pickle.load(f)
        f.close()
        np.random.set_state(seed)

    def save_seed(self, seed_name):
        """Save seed"""
        seed = np.random.get_state()
        f = open(seed_name + '.pckl', 'wb')
        pickle.dump(seed, f)
        f.close()

    def main(self):

        self.iterations = 0
        action = self.candidate_state_evaluator.action_chooser.get_candidate_actions(1)[0]
        self.simulator.choose_active_context(self.iterations)
        self.goal_selector(update_competence=False, attempt=0.0, state_t=self.state_t, state_t1=self.state_t1,
                           reward=False)
        self.simulator.secuencia = 1
        while self.n_epochs <= self.max_epochs:
            if self.n_epochs == 1001:
                self.simulator.secuencia = 2
            if not self.use_goal_selector:
                self.simulator.choose_active_goal(self.iterations)
            # Perception in t (distances, action and motivation)
            self.episode.clean_episode()
            self.episode.set_sensorial_state_t(self.simulator.get_sensorization())
            self.episode.set_action(action)
            self.simulator.apply_action(action)
            # Perception in t+1 (distances and reward)
            self.episode.set_sensorial_state_t1(self.simulator.get_sensorization())
            self.episode.set_reward(self.simulator.get_subgoal_reward())
            # MEMORY MANAGER
            self.memory_manager()
            # Check if the end of trial or epoch has been reached
            if self.evaluating_performance:
                self.check_scenario_restart_performance()
            else:
                self.check_scenario_restart()
                if self.n_epochs % self.epochs_save_data == 0:  # and self.n_epochs > 0:
                    self.evaluating_performance = True
                    self.graph_performance[self.n_epochs] = []
                    self.candidate_state_evaluator.intrinsic_exploration_type = 'Brownian'
                    self.restart_scenario_manually(partially=False)
                else:
                    self.evaluating_performance = False
                    self.candidate_state_evaluator.intrinsic_exploration_type = 'Novelty'
            # MOTIVATION MANAGER
            self.motivation_manager()
            # CANDIDATE STATE EVALUATOR and ACTION CHOOSER
            # Generate new action
            sim_data = self.simulator.get_scenario_data()
            action = self.candidate_state_evaluator.get_action(
                exploration_type=self.active_mot,
                sim_data=sim_data,
                sens_t1=tuple(self.simulator.get_sensorization()),
                corr_sensor=self.corr_sensor,
                corr_type=self.corr_type,
                intrinsic_memory=self.intrinsic_memory.get_contents()
            )
        self.plot_performance()
        self.save_performance_data()
        print("END")
        self.graph1.append((self.state_t1, self.n_epochs, self.n_trials))


    def check_scenario_restart(self):
        end_epoch = False
        end_trial = False
        self.it_trial += 1
        if self.it_trial >= self.max_it_trial or self.simulator.get_goals_state() != self.state_t or self.end_trial:  #Negative trace
            end_trial = True
            self.n_trials += 1
            self.it_trial = 0
            self.end_trial = False
        if self.n_trials >= self.max_trials or self.simulator.get_reward():
            end_epoch = True
            self.n_epochs += 1
            self.n_trials = 0
            self.it_trial = 0
        if end_epoch or end_trial:
            competence = True
            if self.simulator.get_goals_state() != self.state_t:
                if self.active_mot == 'Int':
                    attempt = 0.5 * self.episode.get_reward()  # 0.5*competence
                else:
                    attempt = 1.0 * self.episode.get_reward()  # 1.0*competence
            else:
                attempt = 0.0
                if not self.state_t in self.surs_manager.surs[self.active_corr].context_list:
                    competence = False  # Only update competence if the goal is reachable
            self.restart_scenario(update_competence=competence, attempt=attempt, partially=not end_epoch)
        # OTHERS
        self.debug_print()
        self.iter_min += 1
        self.iterations += 1

    def restart_scenario(self, update_competence, attempt, partially):
        self.state_t1 = self.simulator.get_goals_state()  # End of the trial
        self.graph1.append((self.state_t1, self.n_epochs, self.n_trials))
        if partially:
            # I must save the reward before restarting the scenario
            reward = self.simulator.get_reward()
            self.simulator.restart_scenario_partially()
            self.subgoal_selector(update_competence=update_competence, attempt=attempt, state_t=self.state_t,
                                  state_t1=self.state_t1, reward=reward)
        else:
            self.save_temperatures()  # In order to plot data
            reward = self.simulator.get_reward()
            self.simulator.restart_scenario()  # Restart scenario
            self.goal_selector(update_competence=update_competence, attempt=attempt, state_t=self.state_t,
                               state_t1=self.state_t1, reward=reward)
        self.state_t = self.simulator.get_goals_state()  # Start of the trial
        # ----------------
        # Check if a new correlation is needed or established (In case there is a goal change)
        self.surs_manager.new_sur(self.simulator.active_subgoal, self.simulator.active_context)
        # ----------------
        self.active_corr = self.surs_manager.get_active_sur(
            self.simulator.get_sensorization(), self.simulator.active_subgoal, self.simulator.active_context)
        self.it_blind = 0
        self.reinitialize_memories()
        self.memory_vf.remove_all()

        self.use_motiv_manager = True

    def check_scenario_restart_performance(self):
        print("---EVALUATING PERFORMANCE---")
        end_epoch = False
        end_trial = False
        self.it_trial += 1
        if self.it_trial >= self.max_it_trial or self.simulator.get_goals_state() != self.state_t or self.end_trial:  #Negative trace
            end_trial = True
            self.n_trials += 1
            self.it_trial = 0
            self.end_trial = False
        if self.n_trials >= self.max_trials or self.simulator.get_reward():
            end_epoch = True
            self.epochs_performance += 1
            self.graph_performance[self.n_epochs].append(
                (self.goal_manager.current_goal, self.simulator.get_reward(), self.n_trials))
            self.n_trials = 0
            self.it_trial = 0
            if self.epochs_performance >= self.max_epochs_performance:
                self.evaluating_performance = False
                self.n_epochs += 1
                self.epochs_performance = 0
        if end_epoch or end_trial:
            if self.evaluating_performance:
                self.restart_scenario_manually(partially=not end_epoch)
            else:
                self.restart_scenario(update_competence=False, attempt=0.0, partially=False)
        # OTHERS
        self.debug_print()

    def restart_scenario_manually(self, partially):
        self.state_t1 = self.simulator.get_goals_state()  # End of the trial
        if partially:
            self.simulator.restart_scenario_partially()
            goal_competence = np.sum(self.goal_manager.goals_competence_buffer[self.goal_manager.current_goal][
                                         self.simulator.active_context].get_contents()) / len(
                self.goal_manager.goals_competence_buffer[self.goal_manager.current_goal][
                    self.simulator.active_context].get_contents())
            self.sub_goal_manager.goal_selector(goal_id=self.goal_manager.current_goal,
                                                state=self.simulator.get_goals_state(), goal_competence=goal_competence)
            self.simulator.set_active_subgoal(self.sub_goal_manager.current_goal)
        else:
            self.simulator.restart_scenario()  # Restart scenario
            active_goal = self.goal_manager.goals_list[
                int(self.epochs_performance // (self.max_epochs_performance / len(self.goal_manager.goals_list)))]
            self.goal_manager.current_goal = active_goal
            self.simulator.set_active_goal(self.goal_manager.current_goal)
            goal_competence = np.sum(self.goal_manager.goals_competence_buffer[self.goal_manager.current_goal][
                                         self.simulator.active_context].get_contents()) / len(
                self.goal_manager.goals_competence_buffer[self.goal_manager.current_goal][
                    self.simulator.active_context].get_contents())
            self.sub_goal_manager.goal_selector(goal_id=self.goal_manager.current_goal,
                                                state=self.simulator.get_goals_state(), goal_competence=goal_competence)
            self.simulator.set_active_subgoal(self.sub_goal_manager.current_goal)
        self.state_t = self.simulator.get_goals_state()  # Start of the trial
        # ----------------
        # Check if a new correlation is needed or established (In case there is a goal change)
        self.surs_manager.new_sur(self.simulator.active_subgoal, self.simulator.active_context)
        # ----------------
        self.active_corr = self.surs_manager.get_active_sur(
            self.simulator.get_sensorization(), self.simulator.active_subgoal, self.simulator.active_context)
        self.reinitialize_memories()
        self.memory_vf.remove_all()
        self.use_motiv_manager = True

    def write_logs(self):
        logging.debug('%s  -  %s  -  %s  -  %s  -  %s  -  %s', self.iterations, self.active_mot, self.active_corr,
                      self.corr_sensor, self.corr_type, self.episode.get_episode())

    def debug_print(self):
        if not self.evaluating_performance:
            print("Number of epochs: ", self.n_epochs)

    def reinitialize_memories(self):
        self.traces_buffer.remove_all()  # Reinitialize traces buffer
        self.iter_min = 0
        self.intrinsic_memory.remove_all()  # Reinitialize intrinsic memory
        self.intrinsic_memory.add_episode(self.episode.get_sensorial_state_t1())

    def goal_selector(self, update_competence, attempt, state_t, state_t1, reward):
        if self.use_goal_selector:
            if update_competence:
                self.sub_goal_manager.update_competence(
                    goal_id=self.goal_manager.current_goal,
                    sub_goal_id=self.sub_goal_manager.current_goal,
                    attempt_result=reward * attempt,
                    goals_state_t=state_t,
                    goal_state_t1=state_t1)
                self.goal_manager.update_competence(
                    goal_id=self.sub_goal_manager.current_goal,
                    attempt_result=attempt,
                    context=self.simulator.active_context)
                self.simulator.choose_active_context(self.iterations)
            prob, goal, q_goals = self.goal_manager.goal_selector(context=self.simulator.active_context)
            self.graph2.append((prob, goal, q_goals))
            self.simulator.set_active_goal(self.goal_manager.current_goal)
            goal_competence = np.sum(self.goal_manager.goals_competence_buffer[self.goal_manager.current_goal][
                                         self.simulator.active_context].get_contents()) / len(
                self.goal_manager.goals_competence_buffer[self.goal_manager.current_goal][
                    self.simulator.active_context].get_contents())
            self.sub_goal_manager.goal_selector(goal_id=self.goal_manager.current_goal,
                                                state=self.simulator.get_goals_state(), goal_competence=goal_competence)
            self.simulator.set_active_subgoal(self.sub_goal_manager.current_goal)
            self.graph3.append((self.sub_goal_manager.current_goal, self.goal_manager.current_goal))

    def subgoal_selector(self, update_competence, attempt, state_t, state_t1, reward):
        if self.use_goal_selector:
            if update_competence:
                self.sub_goal_manager.update_competence(
                    goal_id=self.goal_manager.current_goal,
                    sub_goal_id=self.sub_goal_manager.current_goal,
                    attempt_result=reward * attempt,
                    goals_state_t=state_t,
                    goal_state_t1=state_t1)
                self.goal_manager.update_competence(
                    goal_id=self.sub_goal_manager.current_goal,
                    attempt_result=attempt,
                    context=self.simulator.active_context)
            goal_competence = np.sum(self.goal_manager.goals_competence_buffer[self.goal_manager.current_goal][
                                         self.simulator.active_context].get_contents()) / len(
                self.goal_manager.goals_competence_buffer[self.goal_manager.current_goal][
                    self.simulator.active_context].get_contents())
            self.sub_goal_manager.goal_selector(goal_id=self.goal_manager.current_goal,
                                                state=self.simulator.get_goals_state(), goal_competence=goal_competence)
            self.simulator.set_active_subgoal(self.sub_goal_manager.current_goal)
            self.graph3.append((self.sub_goal_manager.current_goal, self.goal_manager.current_goal))

    def motivation_manager(self):
        if self.use_motiv_manager:
            self.corr_sensor, self.corr_type = self.surs_manager.get_active_correlation(
                p=tuple(self.simulator.get_sensorization()),
                active_corr=self.active_corr,
                active_goal=self.simulator.active_subgoal,
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
        self.surs_manager.new_sur(self.simulator.active_subgoal, self.simulator.active_context)
        if not self.surs_manager.surs[self.active_corr].i_reward_assigned:
            self.surs_manager.assign_reward_assigner(self.active_corr, self.episode.get_sensorial_state_t1(),
                                                     self.simulator.active_subgoal, self.simulator.active_context)
        # Memory Manager (Traces, weak traces and antitraces)
        if self.active_mot == 'Int':
            self.it_blind += 1
            self.use_motiv_manager = True
            # If there is a reward, realise reward assignment and save trace in Traces Memory
            if self.episode.get_reward() and not self.evaluating_performance:
                ###
                if not self.surs_manager.surs[self.active_corr].i_reward_assigned:
                    self.surs_manager.assign_reward_assigner(self.active_corr,
                                                             self.episode.get_sensorial_state_t1(),
                                                             self.simulator.active_subgoal,
                                                             self.simulator.active_context,
                                                             1)
                ###
                self.surs_manager.surs[self.active_corr].correlation_evaluator(
                    trace=self.traces_buffer.get_trace(),
                    goals_state=self.state_t
                )  # Here weak trace is already saved
                self.traces_memory_vf.add_traces(self.memory_vf.get_trace_reward())
            elif self.surs_manager.get_reward(self.active_corr, self.simulator.get_subgoal_reward(),
                                              tuple(self.episode.get_sensorial_state_t1()),
                                              self.simulator.active_subgoal, self.simulator.active_context):
                if not self.evaluating_performance:
                    self.surs_manager.surs[self.active_corr].correlation_evaluator(
                        trace=self.traces_buffer.get_trace(),
                        goals_state=self.state_t
                    )
                # The active correlation is now the correlation that has provided the reward
                self.active_corr = self.surs_manager.surs[self.active_corr].i_reward
                self.reinitialize_memories()
        elif self.active_mot == 'Ext':
            self.use_motiv_manager = False
            if self.episode.get_reward():
                # Save as trace in traces_memory of the correlated sensor
                if not self.evaluating_performance:
                    self.surs_manager.surs[self.active_corr].add_trace(
                        trace=self.traces_buffer.get_trace(),
                        sensor=self.corr_sensor,
                        corr_type=self.corr_type,
                        goals_state=self.state_t
                    )
                    self.traces_memory_vf.add_traces(self.memory_vf.get_trace_reward())
                self.use_motiv_manager = True
            elif self.surs_manager.get_reward(self.active_corr, self.simulator.get_subgoal_reward(),
                                              tuple(self.episode.get_sensorial_state_t1()),
                                              self.simulator.active_subgoal, self.simulator.active_context):
                # Save as trace in traces_memory of the correlated sensor
                if not self.evaluating_performance:
                    self.surs_manager.surs[self.active_corr].add_trace(self.traces_buffer.get_trace(),
                                                                       self.corr_sensor, self.corr_type, self.state_t)
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
                        # Save negative trace in the corresponding sensor and start loop again
                        if not self.evaluating_performance:
                            self.surs_manager.surs[self.active_corr].add_antitrace(
                                trace=self.traces_buffer.get_trace(),
                                sensor=self.corr_sensor,
                                corr_type=self.corr_type,
                                goals_state=self.state_t)
                        # ---
                        self.end_trial = True
                        self.use_motiv_manager = True

    @staticmethod
    def calc_sma(data, sma_period):
        j = next(i for i, x in enumerate(data) if x is not None)
        our_range = range(len(data))[j + sma_period - 1:]
        empty_list = [None] * (j + sma_period - 1)
        sub_result = [np.mean(data[i - sma_period + 1: i + 1]) for i in our_range]
        return list(empty_list + sub_result)

    def save_temperatures(self):
        temps = []
        for goal in self.goal_manager.goals_list:
            competence = np.sum(
                self.goal_manager.goals_competence_buffer[goal][self.simulator.active_context].get_contents()) / len(
                self.goal_manager.goals_competence_buffer[goal][
                    self.simulator.active_context].get_contents())
            temp = self.sub_goal_manager.denormalize_value(value=(1 - competence), max_value=0.6, min_value=0.01)
            temps.append(temp)
        self.graph4.append(temps)

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
        plt.yticks([1, 2, 3, 4, 5, 6, 7, 8],
                   ['Ball 1', 'Ball 2', 'Ball 3', 'Ball 4', 'Ball 5', 'Ball 6', 'Ball 7', 'Ball 8'],
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
        # p7 = []
        # p8 = []
        q1 = []
        q2 = []
        q3 = []
        q4 = []
        q5 = []
        q6 = []
        # q7 = []
        # q8 = []
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
            # p7.append(self.graph2[i][0][6])
            # p8.append(self.graph2[i][0][7])
            # Goal q values
            q1.append(self.graph2[i][2][0])
            q2.append(self.graph2[i][2][1])
            q3.append(self.graph2[i][2][2])
            q4.append(self.graph2[i][2][3])
            q5.append(self.graph2[i][2][4])
            q6.append(self.graph2[i][2][5])
            # q7.append(self.graph2[i][2][6])
            # q8.append(self.graph2[i][2][7])
        epochs = list(range(len(self.graph2)))
        ax.plot(epochs, p1, marker='.', color='red', linewidth=0.5, label='Goal1')
        ax.plot(epochs, p2, marker='.', color='green', linewidth=0.5, label='Goal2')
        ax.plot(epochs, p3, marker='.', color='blue', linewidth=0.5, label='Goal3')
        ax.plot(epochs, p4, marker='.', color='orange', linewidth=0.5, label='Goal4')
        ax.plot(epochs, p5, marker='.', color='purple', linewidth=0.5, label='Goal5')
        ax.plot(epochs, p6, marker='.', color='cyan', linewidth=0.5, label='Goal6')
        # ax.plot(epochs, p7, marker='.', color='gold', linewidth=0.5, label='Goal7')
        # ax.plot(epochs, p8, marker='.', color='grey', linewidth=0.5, label='Goal8')
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
        # ax.plot(epochs, q7, marker='.', color='gold', linewidth=0.5, label='Goal7')
        # ax.plot(epochs, q8, marker='.', color='grey', linewidth=0.5, label='Goal8')
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
                 'ball_6': ['cyan', 6],
                 # 'ball_7': ['gold', 7],
                 # 'ball_8': ['grey', 8]
                 }
        line = []
        for i in range(len(self.graph2)):
            ax.plot(i, cdict[self.graph2[i][1]][1], 'ro', color=cdict[self.graph2[i][1]][0])
            line.append(cdict[self.graph2[i][1]][1])
        ax.plot(epochs, line, linewidth=0.5, color='grey', linestyle='--')
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

        # Graph4 temperatures
        t1 = []
        t2 = []
        t3 = []
        t4 = []
        t5 = []
        t6 = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(self.graph4)):
            # Reorganize values to plot thgrem
            t1.append(self.graph4[i][0])
            t2.append(self.graph4[i][1])
            t3.append(self.graph4[i][2])
            t4.append(self.graph4[i][3])
            t5.append(self.graph4[i][4])
            t6.append(self.graph4[i][5])
        epochs = list(range(len(self.graph4)))
        ax.plot(epochs, t1, marker='.', color='orange', linewidth=0.5, label='Goal1')
        ax.plot(epochs, t2, marker='.', color='green', linewidth=0.5, label='Goal2')
        ax.plot(epochs, t3, marker='.', color='blue', linewidth=0.5, label='Goal3')
        ax.plot(epochs, t4, marker='.', color='red', linewidth=0.5, label='Goal4')
        ax.plot(epochs, t5, marker='.', color='cyan', linewidth=0.5, label='Goal5')
        ax.plot(epochs, t6, marker='.', color='purple', linewidth=0.5, label='Goal6')
        # ax.plot(epochs, p7, marker='.', color='gold', linewidth=0.5, label='Goal7')
        # ax.plot(epochs, p8, marker='.', color='grey', linewidth=0.5, label='Goal8')
        plt.legend()
        ax.grid()
        ax.set_xlabel('Epochs', size=12.0)
        ax.set_ylabel('Temperature values', size=12.0)
        ax.set_title('Sub-Goal selector temperatures', size=12.0)
        plt.show()

    def plot_performance(self):
        data_performance = {}
        data_performance['ball_1'] = []
        data_performance['ball_2'] = []
        data_performance['ball_3'] = []
        data_performance['ball_4'] = []
        data_performance['ball_5'] = []
        data_performance['ball_6'] = []
        # data_performance['ball_7'] = []
        # data_performance['ball_8'] = []
        data_it_goal = {}
        data_it_goal['ball_1'] = []
        data_it_goal['ball_2'] = []
        data_it_goal['ball_3'] = []
        data_it_goal['ball_4'] = []
        data_it_goal['ball_5'] = []
        data_it_goal['ball_6'] = []
        # data_it_goal['ball_7'] = []
        # data_it_goal['ball_8'] = []
        for key in sorted(self.graph_performance.keys()):
            epoch_data = self.graph_performance[key]
            data_performance['ball_1'].append(sum(list(zip(*epoch_data[0:10]))[1]))
            data_performance['ball_2'].append(sum(list(zip(*epoch_data[10:20]))[1]))
            data_performance['ball_3'].append(sum(list(zip(*epoch_data[20:30]))[1]))
            data_performance['ball_4'].append(sum(list(zip(*epoch_data[30:40]))[1]))
            data_performance['ball_5'].append(sum(list(zip(*epoch_data[40:50]))[1]))
            data_performance['ball_6'].append(sum(list(zip(*epoch_data[50:60]))[1]))
            # data_performance['ball_7'].append(sum(list(zip(*epoch_data[60:70]))[1]))
            # data_performance['ball_8'].append(sum(list(zip(*epoch_data[70:80]))[1]))
            data_it_goal['ball_1'].append(sum(list(zip(*epoch_data[0:10]))[2]))
            data_it_goal['ball_2'].append(sum(list(zip(*epoch_data[10:20]))[2]))
            data_it_goal['ball_3'].append(sum(list(zip(*epoch_data[20:30]))[2]))
            data_it_goal['ball_4'].append(sum(list(zip(*epoch_data[30:40]))[2]))
            data_it_goal['ball_5'].append(sum(list(zip(*epoch_data[40:50]))[2]))
            data_it_goal['ball_6'].append(sum(list(zip(*epoch_data[50:60]))[2]))
            # data_it_goal['ball_7'].append(sum(list(zip(*epoch_data[60:70]))[2]))
            # data_it_goal['ball_8'].append(sum(list(zip(*epoch_data[70:80]))[2]))

        # Plot performance: para ver si se aprenden los modelos
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data_performance['ball_1'], color='red', label='Goal 1')
        ax.plot(data_performance['ball_2'], color='green', label='Goal 2')
        ax.plot(data_performance['ball_3'], color='blue', label='Goal 3')
        ax.plot(data_performance['ball_4'], color='orange', label='Goal 4')
        ax.plot(data_performance['ball_5'], color='purple', label='Goal 5')
        ax.plot(data_performance['ball_6'], color='cyan', label='Goal 6')
        # ax.plot(data_performance['ball_7'], color='gold', label='Goal 7')
        # ax.plot(data_performance['ball_8'], color='grey', label='Goal 8')
        plt.legend()
        ax.grid()
        ax.set_title('H-GRAIL')
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Performance', size=12)
        plt.xticks(list(range(len(data_performance['ball_1']))), sorted(self.graph_performance.keys()))
        plt.savefig(self.folder + self.filename + "_performance.png", dpi=200)

        # Plot iterations: Para ver si aprende el Q-learning
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
        # media_ball_7 = [x / 10.0 for x in data_it_goal['ball_7']]
        # ax.plot(media_ball_7, color='gold', label='Goal 7')
        # media_ball_8 = [x / 10.0 for x in data_it_goal['ball_8']]
        # ax.plot(media_ball_8, color='grey', label='Goal 8')
        plt.legend()
        ax.grid()
        ax.set_title('H-GRAIL')
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Average trials to reach goal', size=12)
        plt.xticks(list(range(len(media_ball_1))), sorted(self.graph_performance.keys()))
        plt.savefig(self.folder + self.filename + "_trials_goal.png", dpi=200)

    def save_performance_data(self):
        f = open(self.folder + self.filename + '.pckl', 'wb')
        pickle.dump(self.graph_performance, f)
        f.close()

    def load_performance_data(self):
        folder = "performance_data/"
        filename = "h_grail_performance_19"
        f = open(folder + filename + '.pckl', 'rb')
        self.graph_performance = pickle.load(f)
        f.close()


def main():
    instance = MOTIVEN_HGRAIL()
    instance.run()


if __name__ == '__main__':
    main()
