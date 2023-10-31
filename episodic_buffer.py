class EpisodicBuffer(object):
    """
    Class that represents an Episodic Buffer.

    This Episodic buffer (EB) stores the last episodes experienced by the robot.
    The EB has a limited capacity according to the temporal nature of the STM.
    All episodes of the same trace are kept together, so each time a new trace is
    acquired, some of the old ones could be deleted following a FIFO policy.
    This class implements different methods to get/set the buffer size, get its
    contents, add/remove episodes and check if the buffer is full. It also assigns
    the reward when necessary.
    """

    def __init__(self, max_size=10):
        self.buffer = []
        self.max_size = max_size

    def set_max_size(self, max_size):
        self.max_size = max_size

    def get_max_size(self):
        return self.max_size

    def get_size(self):
        return len(self.buffer)

    def get_contents(self):
        return self.buffer

    def is_full(self):
        return self.get_size() >= self.max_size

    def remove_all(self):
        del self.buffer[:]

    def add_episode(self, episode):
        if self.is_full():
            self.buffer.pop(0)
        self.buffer.append(episode)

    def remove_episode(self, index):
        self.buffer.pop(index)
