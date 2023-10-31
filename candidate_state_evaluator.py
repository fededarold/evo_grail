from scipy.spatial import distance

from action_chooser import ActionChooser
'''FEDE: is he importing backprop just to fetch numpy np? I think so'''
import numpy as np
# from back_prop import *
from forward_model import ForwardModel

class CandidateStateEvaluator(object):
    def __init__(self):
        self.forward_model = ForwardModel()
        self.action_chooser = ActionChooser()

        # Variables to control the Brownian motion (intrinsic motivation)
        self.n_random_steps = 0
        self.max_random_steps = 3

        self.intrinsic_exploration_type = 'Novelty'  # 'Brownian' or 'Novelty'

        self.n = 0.5  # Coefficient that regulates the balance between the relevance of distant and near states

    def get_evaluation(self, candidates, corr_sens, tipo, sim_data, sens_t):
        """Return the list os candidates actions sorted according to their value

        :param candidates: list o candidate actions
        :param corr_sens: number of the correlated sensor. 1 - sensor 1, 2 - sensor 2 ... n-sensor n
        :param tipo: type of the correlation: positive ('pos') or negative ('neg')
        :param sim_data: data from the simulator needed to adjust the ForwardModel (baxter_pos, ball_pos, ball_situation, box_pos)
        :param sens_t: actual sensorization to calculate the valuation
        :return: list of candidates actions with its valuation according to the active correlation
        """

        evaluated_candidates = []
        for i in range(len(candidates)):
            valuation = self.get_valuation(candidates[i], corr_sens, tipo, sim_data, sens_t)
            evaluated_candidates.append((candidates[i],) + (valuation,))

        # Order evaluated states
        evaluated_candidates.sort(key=lambda x: x[-1])

        return evaluated_candidates

    def get_valuation(self, candidate, sensor, tipo, sim_data, sens_t):
        """Return the valuation for each individual candidate

        :param candidate: candidate action to evaluate
        :param sensor:  number of the correlated sensor. 1 - sensor 1, 2 - sensor 2 ... n-sensor n
        :param tipo: type of the correlation: positive ('pos') or negative ('neg')
        :param sim_data: data from the simulator needed to adjust the ForwardModel (baxter_pos, ball_pos, ball_situation, box_pos)
        :param sens_t:  actual sensorization to calculate the valuation
        :return: valuation of the candidate state
        """
        sens_t1 = self.forward_model.predicted_state(candidate, sim_data)
        if tipo == 'pos':
            valuation = sens_t1[sensor - 1] - sens_t[sensor - 1]
        elif tipo == 'neg':
            valuation = sens_t[sensor - 1] - sens_t1[sensor - 1]

        return valuation

    def get_action(self, exploration_type, sim_data, sens_t1, corr_sensor, corr_type, intrinsic_memory):

        if exploration_type == 'Int':  # Intrinsic Motivation
            if self.intrinsic_exploration_type == 'Brownian':
                # Brownian motion
                self.n_random_steps += 1
                if self.n_random_steps > self.max_random_steps:
                    action = np.random.uniform(-45, 45)
                    self.max_random_steps = np.random.randint(1, 4)
                    self.n_random_steps = 0
                else:
                    action = 0
            elif self.intrinsic_exploration_type == 'Novelty':
                # action = 0
                candidate_actions = self.action_chooser.get_candidate_actions()
                candidates_eval = self.get_novelty_evaluation(candidate_actions, intrinsic_memory, sim_data)
                action = self.action_chooser.choose_action(candidates_eval)
            elif self.intrinsic_exploration_type == 'Interest':  # Intrinsic Motivation based on interest
                candidate_actions = self.action_chooser.get_candidate_actions()
                candidates_eval = self.get_evaluation(candidate_actions, corr_sensor, corr_type, sim_data,
                                                      sens_t1)
                action = self.action_chooser.choose_action(candidates_eval)

        else:  # Extrinsic motivation -> SUR
            candidate_actions = self.action_chooser.get_candidate_actions()
            candidates_eval = self.get_evaluation(candidate_actions, corr_sensor, corr_type, sim_data,
                                                  sens_t1)
            action = self.action_chooser.choose_action(candidates_eval)

        return action

    def get_novelty_evaluation(self, candidates, trajectory_buffer, sim_data):
        """Return the list of candidates actions sorted according to their novelty value

        :param sim_data: 
        :param candidates: list o candidate actions
        :param trajectory_buffer: buffer that stores the last perceptual states the robot has experienced
        :return: list of candidates actions sorted according to its novelty valuation
        """

        evaluated_candidates = []
        for i in range(len(candidates)):
            valuation = self.get_novelty(candidates[i], trajectory_buffer, sim_data)
            evaluated_candidates.append((candidates[i],) + (valuation,))

        # Order evaluated states
        evaluated_candidates.sort(key=lambda x: x[-1])

        return evaluated_candidates

    def get_novelty(self, candidate_action, trajectory_buffer, sim_data):
        """Return the novelty for each individual candidate

        :param sim_data: 
        :param candidate_action: 
        :param trajectory_buffer: buffer that stores the last perceptual states the robot has experienced
        :return: novelty of the candidate state
        """

        candidate_state = self.forward_model.predicted_state(candidate_action, sim_data)
        novelty = 0
        for i in range(len(trajectory_buffer)):
            novelty += pow(distance.euclidean(candidate_state, trajectory_buffer[i]), self.n)

        novelty = novelty / len(trajectory_buffer)

        return novelty
