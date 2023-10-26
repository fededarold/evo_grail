from sur import SUR


class SURsManager(object):
    """
    Class that represents the SUR Manager module.

    This module identifies when new correlations are needed and contains the set of existing correlations.
    It contains a list with all the existing SUR.
    It also chooses the active coesrelation and gives the value of the reward based on this active correlation.
    """

    def __init__(self, n_sensors):

        self.surs = []
        self.threshold = 0.1  # Threshold to know when to give reward to the sub-correlations
        self.max_surs = 50  # Maximum number of SURs allowed

        self.n_sensors = n_sensors  # Number of sensors. Useful to know how many possible correlations there are

    def new_sur(self, active_goal, context):
        """ This method decides when a new SUR has to be created. Two conditions are considered to do it:
        1- There are no SURs associated with the active goal.
        2- All the SURs associated with the active goal are established.

        :return:
        """
        surs_asoc_active_goal = 0  # Used to count the number of SURs associated with the active goal and context. Condition 1
        index_last_sur_asoc = None  # Used to save the index of the last SUR associated wiht the active goal and context. Condition 2
        for i in range(len(self.surs)):
            if self.surs[i].goal == active_goal and self.surs[i].context == context:
                surs_asoc_active_goal += 1
                index_last_sur_asoc = i

        if surs_asoc_active_goal == 0:  # Condition 1
            self.surs.append(SUR(None, active_goal, context, n_sensors=self.n_sensors))
        elif self.surs[index_last_sur_asoc].established:  # Condition 2
            self.surs.append(SUR(None, active_goal, context, n_sensors=self.n_sensors))

    def get_active_correlation(self, p, active_corr, active_goal, context, goals_state=None):
        """ This method provides the active correlation among all the possible correlations for a given point p

        :return: active_correlation
        """
        corr_sensor, corr_type = self.surs[active_corr].get_active_correlation(p, active_goal, context, goals_state)

        return corr_sensor, corr_type

    def get_active_sur(self, p, active_goal, context):

        for i in range(len(self.surs)):
            if self.surs[i].goal == active_goal and self.surs[i].context == context:
                active_sur = i

        max_certainty = 0
        for i in range(len(self.surs)):
            certainty = self.surs[i].get_certainty(p, active_goal, context)
            if certainty > max_certainty:
                max_certainty = certainty
                active_sur = i

        return active_sur

    def get_reward(self, active_corr, simulator, p, active_goal, context):
        """This method is in charge of provide reward if required
        :param: active_corr: index of the active correlation needed to know who is providing its reward
        :return: reward
        """
        i_r = self.surs[active_corr].i_reward

        if i_r is None:
            reward = simulator
        elif self.surs[i_r].get_certainty(p, active_goal, context) > self.threshold:
            reward = 1
        else:
            reward = 0

        return reward

    def assign_reward_assigner(self, active_corr, p, active_goal, context, scenario=0):
        if scenario:
            self.surs[active_corr].i_reward = None
            self.surs[active_corr].i_reward_assigned = True
        else:
            for i in range(len(self.surs[:active_corr])):
                if self.surs[i].get_certainty(p, active_goal, context) > self.threshold:
                    self.surs[active_corr].i_reward = i
                    self.surs[active_corr].i_reward_assigned = True
                    break
