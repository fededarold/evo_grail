import itertools

import numpy as np

from episodic_buffer import *


class SubGoalManagerMDP(object):
    """
    Class that represents the Goal Manager module.

    This module stores the goals that have been discovered.
    It also selects the goal to achieve at each moment of time.
    This goal selection is based on the competence of the system in reaching each of the goals (CB-IMs).
    """

    def __init__(self, reinforcement_signal='competence'):
        self.current_goal = None
        self.goals_list = []
        self.subgoals_reinforcement_buffer = {}
        self.subgoals_reinforcement = {}
        self.PT = 10  # number of attempts on which calculate competence
        self.goal_q_values = {}
        self.reinforcement_signal = reinforcement_signal  # 'competence', 'reward'

    def add_goal(self, goal_id):
        self.goals_list.append(goal_id)
        self.initialise_buffers()

    def initialise_buffers(self):
        for goal in self.goals_list:
            self.subgoals_reinforcement_buffer[goal] = {}
            self.subgoals_reinforcement[goal] = {}
            for subgoal in self.goals_list:
                self.subgoals_reinforcement_buffer[goal][subgoal] = EpisodicBuffer(self.PT * 2)
                self.subgoals_reinforcement_buffer[goal][subgoal].add_episode(0)  # Initialise buffer
                self.subgoals_reinforcement[goal][subgoal] = 0.0
            self.goal_q_values[goal] = self.create_q_table(len(self.goals_list))

    def create_q_table(self, state_dimension):
        states = list(itertools.product([False, True], repeat=state_dimension))
        q_table = {}
        for state in states:
            q_table[state] = {}
            for goal in self.goals_list:
                q_table[state][goal] = 0.0
        return q_table

    def delete_goal(self, goal_id):
        self.goals_list.remove(goal_id)
        self.initialise_buffers()

    def update_competence(self, goal_id, sub_goal_id, attempt_result, goals_state_t, goal_state_t1):
        self.subgoals_reinforcement_buffer[goal_id][sub_goal_id].add_episode(attempt_result)
        self.calculate_competence(goal_id, sub_goal_id)
        self.update_q_values(goal_id, sub_goal_id, state_t=goals_state_t, state_t1=goal_state_t1)

    def calculate_competence(self, goal_id, sub_goal_id):
        """
        Calculate competence-based intrinsic motivation (CB-IM) signal.
        The CB-IM signal is the difference between two averages of competence
        predictions (CP), each one calculated over a period PT of attempts
        (related to the same goal). So that, the two averages cover a period
        of 2PT attempts going backwards from the current selection into the past.
        Before covering the entire period 2PT, the signal C is calculated by
        dividing in two the actual collection of predictions.
        """
        self.subgoals_reinforcement[goal_id][sub_goal_id] = \
            self.subgoals_reinforcement_buffer[goal_id][sub_goal_id].get_contents()[-1]

    def update_q_values(self, goal_id, sub_goal_id, state_t, state_t1, alfa=0.1, gamma=0.6):  # alfa=0.1, gamma=0.3
        """
        At time t and state s_t, the value of each selected unit q(s_t, k_t),
        representing the motivation to select the corresponding goal,
        is updated through a Q-learning  algorithm maximising the intrinsic
        reinforcement (ir(k) -> CB-IM signal) for the goal associated to that unit
        q(s_t, k_t) = q(s_t, k_t) + alfa * (
            ir(k_{t+1}) + gamma * max[q(s_{t+1}, k_{t+1})] - q(s_t, k_t)
            )
        """
        max_sub_goal = max(self.goal_q_values[goal_id][state_t1], key=self.goal_q_values[goal_id][state_t1].get)
        self.goal_q_values[goal_id][state_t][sub_goal_id] = self.goal_q_values[goal_id][state_t][sub_goal_id] + alfa * (
                self.subgoals_reinforcement[goal_id][sub_goal_id] + gamma *
                self.goal_q_values[goal_id][state_t1][
                    max_sub_goal] - self.goal_q_values[goal_id][state_t][sub_goal_id])

    def goal_selector(self, goal_id, state, goal_competence):
        """ Selection of goals on the basis of competence-based IMs (CB-IMs)"""
        # to avoid choosing goals that are already active
        goals_list = self.goals_list[:]
        if state[0]:
            goals_list.remove('ball_1')
        if state[1]:
            goals_list.remove('ball_2')
        if state[2]:
            goals_list.remove('ball_3')
        if state[3]:
            goals_list.remove('ball_4')
        if state[4]:
            goals_list.remove('ball_5')
        if state[5]:
            goals_list.remove('ball_6')
        # if state[6]:
        #     goals_list.remove('ball_7')
        # if state[7]:
        #     goals_list.remove('ball_8')

        q_values = []
        for goal in goals_list:
            q_values.append(self.goal_q_values[goal_id][state][goal])
        probabilities = self.softmax(np.array(q_values),
                                     goal_competence)  # probabilities = self.softmax(np.array(q_values))
        self.current_goal = np.random.choice(goals_list, p=probabilities)

    # @staticmethod
    def softmax(self, x, competence, temp=0.01):  # 0.02
        """Compute softmax values for each sets of scores in x."""
        # temp = self.denormalize_value(value=(1-competence), max_value=0.2, min_value=0.01)
        temp = 0.6  # self.denormalize_value(value=(1-competence), max_value=0.6, min_value=0.01)
        return np.exp(x / temp) / np.sum(np.exp(x / temp), axis=0)

    @staticmethod
    def normalize_value(value, max_value, min_value=0.0):
        return (value - min_value) / (max_value - min_value)

    @staticmethod
    def denormalize_value(value, max_value, min_value=0.0):
        return value * (max_value - min_value) + min_value
