class TracesMemory(object):
    """
    Class that represents a Memory of Traces.

    A trace is a list of episodes with an assigned value (expected reward)
    which are stored together.
    It distinguish between Positive Traces (named Traces), Negative Traces
    (named AntiTraces) and WeakPositive Traces (named WeakTraces)
    
    This class implements different methods to get/set the different traces
    lists, get their contents and add/remove traces.
    """

    def __init__(self):
        self.traces_list = []
        self.antitraces_list = []
        self.weaktraces_list = []

    def add_traces(self, traces):
        self.traces_list.append(traces)

    def add_antitraces(self, traces):
        self.antitraces_list.append(traces)

    def add_weaktraces(self, traces):
        self.weaktraces_list.append(traces)

    def get_traces_list(self):
        return self.traces_list

    def get_antitraces_list(self):
        return self.antitraces_list

    def get_weaktraces_list(self):
        return self.weaktraces_list
