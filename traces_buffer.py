from episodic_buffer import EpisodicBuffer


class TracesBuffer(EpisodicBuffer):

    def reward_assignment(self):
        reward = 1.0
        for i in reversed(range(len(self.buffer))):
            self.buffer[i][-1] = reward
            reward -= 1.0 / self.max_size

    def pain_assignment(self):
        reward = -1.0
        for i in reversed(range(len(self.buffer))):
            self.buffer[i][-1] = min(0, reward)
            reward -= -1.0 / 5
            # reward -= -1.0 / self.max_size  # reward -= 1.0 / len(self.buffer)

    def get_trace(self):
        """Return the trace values needed to use in the SUR, the sensorization in t+1"""
        trace = []
        for i in range(len(self.buffer)):
            trace.append(self.buffer[i][2])

        return tuple(trace)

    def get_trace_reward(self):
        """Return the trace values and they associated reward to use to train the VF network"""
        self.reward_assignment()
        trace = []
        for i in range(len(self.buffer)):
            trace.append(self.buffer[i][2:])

        return tuple(trace)

    def get_trace_pain(self):
        """Return the trace values and they associated pain to use to train the VF network"""
        self.pain_assignment()
        trace = []
        for i in range(len(self.buffer)):
            trace.append(self.buffer[i][2:])

        return tuple(trace)

    def get_antitrace(self):
        """Return the antitrace values needed to use in the SUR,
        the sensorization in t+1 obtained using the extrinsic motivation"""
        for i in reversed(range(len(self.buffer))):
            if self.buffer[i][4] == 'Int':
                break
        antitrace = self.buffer[i:][2]

        return tuple(antitrace)
