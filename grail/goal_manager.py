import numpy as np

from episodic_buffer import *


class GoalManager(object):
    """
    Class that represents the Goal Manager module.

    This module stores the goals that have been discovered.
    It also selects the goal to achieve at each moment of time.
    This goal selection is based on the competence of the system in reaching each of the goals (CB-IMs).
    """

    def __init__(self):
        
        self.softmax_temp=None
        self.beta_val=None
        
        self.current_goal = None
        self.goals_list = []
        self.goals_competence_buffer = {}
        self.goals_competence = {}
        self.PT = 10  # number of attempts on which calculate competence
        self.q_values = {}
        self.contexts_list = ["c0", "c1"]  # number of possible contexts

    def add_goal(self, goal_id):
        self.goals_list.append(goal_id)
        self.goals_competence_buffer[goal_id] = {}
        self.goals_competence[goal_id] = {}
        self.q_values[goal_id] = {}
        for context in self.contexts_list:
            self.goals_competence_buffer[goal_id][context] = EpisodicBuffer(self.PT * 2)
            self.goals_competence_buffer[goal_id][context].add_episode(0)  # Initialise buffer
            self.goals_competence[goal_id][context] = 0.0
            self.q_values[goal_id][context] = 0.0

    def delete_goal(self, goal_id):
        self.goals_list.remove(goal_id)
        del self.goals_competence_buffer[goal_id]
        del self.goals_competence[goal_id]
        del self.q_values[goal_id]

    def update_competence(self, goal_id, attempt_result, context):
        self.goals_competence_buffer[goal_id][context].add_episode(attempt_result)
        self.calculate_competence(goal_id, context)
        self.update_q_values(goal_id, context)

    def calculate_competence(self, goal_id, context):
        """
        Calculate competence-based intrinsic motivation (CB-IM) signal.
        The CB-IM signal is the difference between two averages of competence
        predictions (CP), each one calculated over a period PT of attempts
        (related to the same goal). So that, the two averages cover a period
        of 2PT attempts going backwards from the current selection into the past.
        Before covering the entire period 2PT, the signal C is calculated by
        dividing in two the actual collection of predictions.
        """
        pt = int(len(self.goals_competence_buffer[goal_id][context].get_contents()) / 2)
        sum1 = sum(self.goals_competence_buffer[goal_id][context].get_contents()[pt:])
        if len(self.goals_competence_buffer[goal_id][context].get_contents()) % 2 == 0:
            sum2 = sum(self.goals_competence_buffer[goal_id][context].get_contents()[:pt])
        else:
            sum2 = sum(self.goals_competence_buffer[goal_id][context].get_contents()[:pt]) + \
                   self.goals_competence_buffer[goal_id][context].get_contents()[0]
        competence = max(0, sum1 / float(pt) - sum2 / float(pt))
        self.goals_competence[goal_id][context] = competence

    def update_q_values(self, goal_id, context): #, beta=0.3):
        """
        The value of each goal g is updated through an EMA of the CB-IM signal
        q(k_t,s_t) = q(k_t, s_t) + beta * (ir(k_{t+1}) - q(k_t, s_t))
        """
        beta = self.beta_val      
        self.q_values[goal_id][context] = self.q_values[goal_id][context] + beta * (
                self.goals_competence[goal_id][context] - self.q_values[goal_id][context])

    def goal_selector(self, context):
        """ Selection of goals on the basis of competence-based IMs (CB-IMs)"""
        q_values = []
        for goal in self.goals_list:
            q_values.append(self.q_values[goal][context])
        probabilities = self.softmax(np.array(q_values))
        self.current_goal = np.random.choice(self.goals_list, p=probabilities)
        return probabilities, self.current_goal, q_values

    #TODO: it was 0.02
    # @staticmethod
    def softmax(self, x): #, temp=self.softmax_temp):  # 0.01
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x / self.softmax_temp) / np.sum(np.exp(x / self.softmax_temp), axis=0)
    
    def set_softmax_temp(self, temp):
        self.softmax_temp = temp
        
    def get_softmax_temp(self):
        return self.softmax_temp
    
    def set_beta_val(self, beta):
        self.beta_val = beta
        
    def get_beta_val(self):
        return self.beta_val
