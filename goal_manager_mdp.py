import itertools

import numpy as np

from episodic_buffer import *


class GoalManagerMDP(object):
    """
    Class that represents the Goal Manager module.

    This module stores the goals that have been discovered.
    It also selects the goal to achieve at each moment of time.
    This goal selection is based on the competence of the system in reaching each of the goals (CB-IMs).
    """

    def __init__(self):
        self.current_goal = None
        self.goals_list = []
        self.goals_competence_buffer = {}
        self.goals_competence = {}
        self.PT = 10  # number of attempts on which calculate competence
        self.q_values = {}

    def add_goal(self, goal_id):
        self.goals_list.append(goal_id)
        self.goals_competence_buffer[goal_id] = EpisodicBuffer(self.PT * 2)
        self.goals_competence_buffer[goal_id].add_episode(0)  # Initialise buffer
        self.goals_competence[goal_id] = 0.0
        self.q_values = self.create_q_table(len(self.goals_list))

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
        del self.goals_competence_buffer[goal_id]
        del self.goals_competence[goal_id]
        self.q_values = self.create_q_table(len(self.goals_list))

    def update_competence(self, goal_id, attempt_result, goals_state_t, goal_state_t1):
        self.goals_competence_buffer[goal_id].add_episode(attempt_result)
        self.calculate_competence(goal_id)
        self.update_q_values(goal_id, state_t=goals_state_t, state_t1=goal_state_t1)

    def calculate_competence(self, goal_id):
        """
        Calculate competence-based intrinsic motivation (CB-IM) signal.
        The CB-IM signal is the difference between two averages of competence
        predictions (CP), each one calculated over a period PT of attempts
        (related to the same goal). So that, the two averages cover a period
        of 2PT attempts going backwards from the current selection into the past.
        Before covering the entire period 2PT, the signal C is calculated by
        dividing in two the actual collection of predictions.
        """
        pt = int(len(self.goals_competence_buffer[goal_id].get_contents()) / 2)
        sum1 = sum(self.goals_competence_buffer[goal_id].get_contents()[pt:])
        if len(self.goals_competence_buffer[goal_id].get_contents()) % 2 == 0:
            sum2 = sum(self.goals_competence_buffer[goal_id].get_contents()[:pt])
        else:
            sum2 = sum(self.goals_competence_buffer[goal_id].get_contents()[:pt]) + \
                   self.goals_competence_buffer[goal_id].get_contents()[0]
        competence = max(0, sum1 / float(pt) - sum2 / float(pt))
        self.goals_competence[goal_id] = competence

    def update_q_values(self, goal_id, state_t, state_t1, alfa=0.1, gamma=0.3):
        """
        At time t and state s_t, the value of each selected unit q(s_t, k_t),
        representing the motivation to select the corresponding goal,
        is updated through a Q-learning  algorithm maximising the intrinsic
        reinforcement (ir(k) -> CB-IM signal) for the goal associated to that unit
        q(s_t, k_t) = q(s_t, k_t) + alfa * (
            ir(k_{t+1}) + gamma * max[q(s_{t+1}, k_{t+1})] - q(s_t, k_t)
            )
        """
        max_goal = max(self.q_values[state_t1], key=self.q_values[state_t1].get)
        self.q_values[state_t][goal_id] = self.q_values[state_t][goal_id] + alfa * (
                self.goals_competence[goal_id] + gamma * self.q_values[state_t1][max_goal] - self.q_values[state_t][
            goal_id])

    def goal_selector(self, state):
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

        q_values = []
        for goal in goals_list:
            q_values.append(self.q_values[state][goal])
        probabilities = self.softmax(np.array(q_values))
        self.current_goal = np.random.choice(goals_list, p=probabilities)
        return probabilities, self.current_goal, q_values

    @staticmethod
    def softmax(x, temp=0.02):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x / temp) / np.sum(np.exp(x / temp), axis=0)
