import math

import numpy as np

from traces_memory import TracesMemory


class DistancesCertainty(object):
    """
    This class represents a mathematical model that defines the implementation used for the creation of certainty maps.
    Its aim is to obtain the certainty value for a point 'p' given.
    """

    def __init__(self, n_sensors):

        self.traces_memory = TracesMemory()
        self.nt = 0  # Number of traces: nt=Nht+cf*Nst: sum of p-traces and cf times w-traces
        self.n_antitraces = 0
        self.number_of_goals_without_antitraces = 0  # To check when a correlation has become established
        self.cf = 0.7  # Reliability factor

        self.l_inf = (0.0,) * n_sensors
        self.l_sup = (1.0,) * n_sensors

        self.nt_factor = 100.0  ##100.0#75.0#100.0#4.0#6.0
        self.k = pow(0.05, 1.0 / (self.nt_factor - 1.0))
        self.ce = 1.0
        self.m = 50.0

        self.cp = 1.0  # 0.6  # 0.6  # Stability factor
        self.cw = 0.3  # 0.3  # Weighting factor for w-traces
        self.ca = 1.0  # Weighting factor for n-traces

        self.epsilon = 100  # 1.0 / 100.0

        self.percentile = 100  # q-th percentile

        self.traces_min_distances_map = ()
        self.weaktraces_min_distances_map = ()
        self.antitraces_min_distances_map = ()

    @staticmethod
    def get_min_distances_map(t):
        """Return the set of the minimum distances for all the points in T.
         T is the set of trace points (episodes) used to define the certainty map."""
        distances_map = [[None] * len(t) for i in range(len(t[0]))]
        n_gr = 99999
        for k in range(len(t)):
            for i in range(len(t[0])):
                d_pos = n_gr
                d_neg = d_pos
                for j in range(len(t)):
                    if k != j:
                        d = t[k][i] - t[j][i]
                        if d > 0:
                            d_pos = min(d_pos, d)
                        else:
                            d_neg = min(d_neg, -d)
                if d_pos > n_gr / 2:
                    d_pos = -1
                if d_neg > n_gr / 2:
                    d_neg = -1
                distances_map[i][k] = max(d_pos, d_neg)
        return distances_map

    @staticmethod
    def get_percentile(y, d):
        """Return the percentile 'y' over the set 'D'"""
        de = np.percentile(d, y, axis=1)
        de = de.tolist()
        return de

    def get_dr(self, t):
        """Return the minimum distances to the sensor limits (distance 
        from each of the n components of each episode contained in T)"""
        dr = []
        for i in range(len(t[0])):
            dist_sup = abs(self.l_sup[i] - self.l_inf[i])
            dist_inf = dist_sup
            for j in range(len(t)):
                dist_sup_tmp = abs(t[j][i] - self.l_sup[i])
                dist_inf_tmp = abs(t[j][i] - self.l_inf[i])
                dist_sup = min(dist_sup, dist_sup_tmp)
                dist_inf = min(dist_inf, dist_inf_tmp)
            dist = max(dist_sup, dist_inf)
            dr.append(dist)
        return dr

    @staticmethod
    def get_h(t, p):
        """Return the distances between each of the n components of the trace points contained in T and any point p"""
        h = [[None] * len(t) for i in range(len(t[0]))]
        for i in range(len(t[0])):
            for j in range(len(t)):
                h[i][j] = abs(p[i] - t[j][i])
        return h

    def get_hlim(self, min_distances_map, percentile, t, n_traces):
        """Return hlim. the limit distances in the m dimensions from 
        which traces quickly decrease their effect on the state space"""
        de = self.get_percentile(percentile, min_distances_map)
        dr = self.get_dr(t)
        hlim = []
        for i in range(len(dr)):
            if dr[i] > de[i]:
                hlim.append((de[i] + (dr[i] - de[i]) * pow(self.k, n_traces - 1)) / 2.0)
            else:
                hlim.append(de[i] / 2.0)
        return hlim

    def get_hn(self, min_distances_map, percentile, t, n_traces, p):
        """Return the effective distances in the m dimensions between the trace points and any point p"""
        h = self.get_h(t, p)
        hlim = self.get_hlim(min_distances_map, percentile, t, n_traces)
        hn = [[None] * len(h) for i in range(len(h[0]))]
        for i in range(len(h)):
            for j in range(len(h[0])):
                if h[i][j] < self.ce * hlim[i]:
                    hn[j][i] = h[i][j]
                else:
                    hn[j][i] = self.ce * hlim[i] + (h[i][j] - self.ce * hlim[i]) * self.m
        return hn, h

    def get_weight(self, min_distances_map, percentile, t, n_traces, p):
        """Return the weights of the trace points in any point p"""
        hn, h = self.get_hn(min_distances_map, percentile, t, n_traces, p)
        w = []
        for i in range(len(hn)):
            norm_value = []
            norm_value_aux = []
            for j in range(len(hn[0])):
                norm_value.append(hn[i][j] / (self.l_sup[j] - self.l_inf[j]))
                norm_value_aux.append(h[j][i] / (self.l_sup[j] - self.l_inf[j]))
            w.append(max(0, 1 - np.linalg.norm(norm_value)) / (np.linalg.norm(norm_value_aux) + 1 / self.epsilon))
        return w

    def get_certainty_value(self, p):
        """Return the certainty value c for a point p combining the weights of
         p-traces(w_positive), n-traces(w_negative) and w-traces(w_weak)"""
        traces_tuple = self.trace_list_to_tuple(self.traces_memory.get_traces_list())
        antitraces_tuple = self.trace_list_to_tuple(self.traces_memory.get_antitraces_list())
        weaktraces_tuple = self.trace_list_to_tuple(self.traces_memory.get_weaktraces_list())
        if traces_tuple == ():
            w_positive = ()
        else:
            w_positive = self.get_weight(self.traces_min_distances_map, self.percentile, traces_tuple, self.nt, p)
        if antitraces_tuple == ():
            w_negative = ()
        else:
            w_negative = self.get_weight(self.antitraces_min_distances_map, self.percentile, antitraces_tuple,
                                         self.n_antitraces, p)
        if weaktraces_tuple == ():
            w_weak = ()
        else:
            w_weak = self.get_weight(self.weaktraces_min_distances_map, self.percentile, weaktraces_tuple, self.nt, p)
        suma = 0
        for i in range(len(w_positive)):
            suma += w_positive[i]
        for i in range(len(w_weak)):
            # suma += self.cw * w_weak[i]
            suma += w_weak[i]
        for i in range(len(w_negative)):
            # suma -= self.ca * w_negative[i]
            suma -= w_negative[i]
        # c = max(0, math.tanh(self.cp * suma))
        c = np.sign(suma) * math.pow(abs(math.tanh(self.cp * suma)), 0.1)

        return c if self.nt > 3.0 else c * 0.05  # I only rely on certainty when I have a certain number of traces.

    def add_traces(self, new_trace):
        self.traces_memory.add_traces(new_trace)
        self.nt += 1
        # if not intrinsic_guided:
        self.number_of_goals_without_antitraces += 1
        # Update traces minimum distances map
        t = self.trace_list_to_tuple(self.traces_memory.get_traces_list())
        self.traces_min_distances_map = self.get_min_distances_map(t)

    def add_weaktraces(self, new_trace):
        self.traces_memory.add_weaktraces(new_trace)
        self.nt += self.cf * 1
        # Update traces minimum distances map
        t = self.trace_list_to_tuple(self.traces_memory.get_weaktraces_list())
        self.weaktraces_min_distances_map = self.get_min_distances_map(t)

    def add_antitraces(self, new_trace, intrinsic_guided=False):
        self.traces_memory.add_antitraces(new_trace)
        self.n_antitraces += 1
        if not intrinsic_guided:
            self.number_of_goals_without_antitraces = max(0, self.number_of_goals_without_antitraces - 1)
        # Update traces minimum distances map
        t = self.trace_list_to_tuple(self.traces_memory.get_antitraces_list())
        self.antitraces_min_distances_map = self.get_min_distances_map(t)

    def get_number_of_goals_without_antitraces(self):
        return self.number_of_goals_without_antitraces

    @staticmethod
    def trace_list_to_tuple(trace_list):
        """ Transform a list into a tuple

        :param trace_list: a list of traces containing episodes (tuples)
        :return: a tuple of traces containing episodes (tuples)
        """
        trace_tuple = ()
        for i in range(len(trace_list)):
            trace_tuple += trace_list[i]
        return trace_tuple
