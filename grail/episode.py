class Episode(object):
    """
    Class that represents an Episode.

    An Episode (E) is a sample of the real world response to the robot actions.
    Within the MDB, an episode is made up of the sensorial state in t, the applied action in t,
    the sensorial state in t + 1 and the reward in t + 1:
    episode = {S(t), A(t), S(t+1), R(t+1)}
    This class implements different methods to get/set the sensorial states, action or reward
    of the episode.
    """

    def __init__(self):
        self.sensorial_state_t = []
        self.action_t = 0
        self.sensorial_state_t1 = []
        self.reward_t1 = 0

    def get_sensorial_state_t(self):
        return self.sensorial_state_t

    def get_action(self):
        return self.action_t

    def get_sensorial_state_t1(self):
        return self.sensorial_state_t1

    def get_reward(self):
        return self.reward_t1

    def get_episode(self):
        return [self.sensorial_state_t, self.action_t, self.sensorial_state_t1, self.reward_t1]  # , self.motivation]

    def set_sensorial_state_t(self, sensorial_state):
        self.sensorial_state_t = sensorial_state

    def set_action(self, action):
        self.action_t = action

    def set_sensorial_state_t1(self, sensorial_state):
        self.sensorial_state_t1 = sensorial_state

    def set_reward(self, reward):
        self.reward_t1 = reward

    def set_episode(self, sensorial_state_t, action, sensorial_state_t1, reward):
        self.sensorial_state_t = sensorial_state_t
        self.action_t = action
        self.sensorial_state_t1 = sensorial_state_t1
        self.reward_t1 = reward

    def clean_episode(self):
        self.sensorial_state_t = []
        self.action_t = 0
        self.sensorial_state_t1 = []
        self.reward_t1 = 0
