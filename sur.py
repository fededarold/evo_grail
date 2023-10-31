from distances_certainty import DistancesCertainty


class SUR(object):
    """
    Class that represents the SUR module.

    This module identifies new correlations and contains the set of existing correlations.
    It contains the Correlation Evaluator that is an algorithm that has to be executed each time a trace is added
    to the Traces Memory and searches for possible correlations to be stored in the Traces Correlation Memory.
    It also has the Distances Certainty module that makes possible the creation of certainty maps using the traces
    stored as positive-traces, negative-traces and weak-traces, which aim is to obtain the certainty value for a
    point p given.
    """

    def __init__(self, reward_assigner, goal_id, context_id, n_sensors):
        self.n_sensor = n_sensors  # Number of sensors. Useful to know how many possible correlations there are
        self.min_ep = 2  # 5  # Minimum number of episodes to consider the correlation possible
        self.same_values_accepted = 1  # Number of sensor admitted to be equal

        # Correlations, Traces Memories and certainty evaluators
        self.candidate_surs = []
        for i in range(self.n_sensor):
            self.candidate_surs.append(
                (DistancesCertainty(self.n_sensor), DistancesCertainty(self.n_sensor)))  # (Sn_pos, Sn_neg)

        self.sur_active = 0  # 1 - Sensor 1, 2 - Sensor 2, ... n- sensor n, 0 - no correlation
        self.sur_type = ''  # 'pos' - Positive correlation, 'neg' - Negative correlation, '' - no correlation
        self.sur_threshold = 0.1  # Threshold to know when to consider Extrinsic Motivation and when Intrinsic

        self.established = False
        self.sur_established = 0
        self.sur_established_type = ''

        self.i_reward = reward_assigner
        self.i_reward_assigned = False
        self.goal = goal_id
        self.context = context_id
        self.context_list = []  # List of known context states.
        # To only allow learning when the goal has been reached once in that state.

        self.tb_max = 10  # 20#8  # Number of goals without antitraces needed to consider the correlation established

    def correlation_evaluator(self, trace, goals_state=None):

        """This method evaluates the possible correlations existing in a trace T and save them in the proper Correlation
        Traces Memory Buffer

        Keyword arguments:
        Trace -- List of tuples, it is a List of episodes-sensorization(tuples)
        """
        if len(trace) >= self.min_ep:
            for i in range(self.n_sensor):
                p_corr = True  # Positive correlation
                n_corr = True  # Negative correlation
                same_value = 0  # Number of times a sensor has the same value in two consecutive episodes
                for j in reversed(range(len(trace) - self.min_ep, len(trace))):
                    if p_corr:  # The case when positive correlation is active
                        if trace[j][i] > trace[j - 1][i]:
                            n_corr = False  # Negative correlation is no longer possible for this sensor
                        elif trace[j][i] < trace[j - 1][i]:
                            p_corr = False  # Positive correlation is no longer possible for this sensor
                        else:  # Trace[j][i]=Trace[j-1][i]
                            same_value += 1
                            if same_value > self.same_values_accepted:
                                n_corr = False
                                p_corr = False
                    elif n_corr:  # The case when negative correlation is active
                        if trace[j][i] > trace[j - 1][i]:
                            n_corr = 0  # Negative correlation is no longer possible for this sensor
                        elif trace[j][i] < trace[j - 1][i]:
                            p_corr = 0  # Positive correlation is no longer possible for this sensor
                        else:  # Trace[j][i]=Trace[j-1][i]
                            same_value += 1
                            if same_value > self.same_values_accepted:
                                n_corr = False
                                p_corr = False
                # If there is a correlation, save it in the pertinent correlation trace memory
                if p_corr:
                    # self.add_weaktrace(trace, i + 1, 'pos')
                    # print("No meto traza positiva")
                    pass
                elif n_corr:
                    self.add_weaktrace(trace, i + 1, 'neg')
                    if not goals_state in self.context_list:
                        self.context_list.append(goals_state)

    def get_active_correlation(self, p, active_goal, context, goals_state=None):
        """
        # Este metodo despues ira dentro del motivation manager, y lo de correlaciones importado alli tambien (dentro de un modulo que sea modelos de utilidad)

        # Evaluo la certeza del nuevo punto en todas las correlaciones para ver si pertenece a alguna
        # Si es mayor que un umbral para alguna de ellas, considero la mayor y si hay empate, una al azar
        # Si es menor que el umbral, consireo la motivacion intrinseca
        :param p:
        :return:
        """
        if active_goal == self.goal and context == self.context:  # and (goals_state in self.context_list):
            certainty_values_list = []
            for sensor in range(self.n_sensor):
                certainty_values_list.append(
                    (self.candidate_surs[sensor][0].get_certainty_value(p),
                     self.candidate_surs[sensor][1].get_certainty_value(p))
                )  # (cn_pos, cn_neg)
            if self.established:
                if self.sur_established_type == 'pos':
                    certainty_value = certainty_values_list[self.sur_established - 1][0]  # cn_pos
                else:
                    certainty_value = certainty_values_list[self.sur_established - 1][1]  # cn_neg
                if self.sur_threshold > certainty_value:
                    self.sur_active = 0
                    self.sur_type = ''
                else:
                    self.sur_active = self.sur_established
                    self.sur_type = self.sur_established_type
            else:
                certainty_values_list_pos, certainty_values_list_neg = zip(*certainty_values_list)
                if self.sur_threshold > max(max(certainty_values_list_pos), max(certainty_values_list_neg)):
                    self.sur_active = 0
                    self.sur_type = ''
                else:
                    if max(certainty_values_list_neg) > max(certainty_values_list_pos):
                        i = certainty_values_list_neg.index(max(certainty_values_list_neg))
                        self.sur_type = 'neg'
                    else:
                        i = certainty_values_list_pos.index(max(certainty_values_list_pos))
                        self.sur_type = 'pos'
                    self.sur_active = i + 1
        else:
            self.sur_active = 0
            self.sur_type = ''
        return self.sur_active, self.sur_type

    def get_certainty(self, p, active_goal, context):
        """This method provides the maximum certainty value of the correlations

        :param p:
        :return:
        """
        if active_goal == self.goal and context == self.context:
            certainty_values_list = []
            for sensor in range(self.n_sensor):
                certainty_values_list.append(
                    (self.candidate_surs[sensor][0].get_certainty_value(p),
                     self.candidate_surs[sensor][1].get_certainty_value(p))
                )  # (cn_pos, cn_neg)
            if self.established:
                if self.sur_established_type == 'pos':
                    certainty_value = certainty_values_list[self.sur_established - 1][0]  # cn_pos
                else:
                    certainty_value = certainty_values_list[self.sur_established - 1][1]  # cn_neg
            else:
                certainty_values_list_pos, certainty_values_list_neg = zip(*certainty_values_list)
                certainty_value = max(max(certainty_values_list_pos), max(certainty_values_list_neg))
        else:
            certainty_value = 0
        return certainty_value

    def add_trace(self, trace, sensor, corr_type, goals_state=None):
        if not goals_state in self.context_list:
            self.context_list.append(goals_state)
        if not self.established:
            for i in reversed(range(len(trace))):
                if corr_type == 'neg':
                    if trace[i][sensor - 1] >= trace[i - 1][sensor - 1]:
                        break
                elif corr_type == 'pos':
                    if trace[i][sensor - 1] <= trace[i - 1][sensor - 1]:
                        break
            if corr_type == 'pos':
                self.candidate_surs[sensor - 1][0].add_traces(trace[i:])
            elif corr_type == 'neg':
                self.candidate_surs[sensor - 1][1].add_traces(trace[i:])
                # Check if the correlation is established (it could only happen after adding a trace)
                self.is_sur_established()
            # print("Trace added in sensor ", sensor, " and type ", corr_type)
            # print("")

    def add_antitrace(self, trace, sensor, corr_type, goals_state=None):
        if goals_state in self.context_list:
            if corr_type == 'pos':
                self.candidate_surs[sensor - 1][0].add_antitraces(trace)
            elif corr_type == 'neg':
                self.candidate_surs[sensor - 1][1].add_antitraces(trace)
            # print("Antitrace added in sensor ", sensor, " and type ", corr_type)
            # print("")

    def add_weaktrace(self, trace, sensor, corr_type):
        if not self.established:
            for i in reversed(range(len(trace))):
                if corr_type == 'neg':
                    if trace[i][sensor - 1] >= trace[i - 1][sensor - 1]:
                        break
                elif corr_type == 'pos':
                    if trace[i][sensor - 1] <= trace[i - 1][sensor - 1]:
                        break
            if corr_type == 'pos':
                self.candidate_surs[sensor - 1][0].add_weaktraces(trace[i:])
            elif corr_type == 'neg':
                self.candidate_surs[sensor - 1][1].add_weaktraces(trace[i:])
            # print("Weak trace added in sensor ", sensor, " and type ", corr_type)
            # print("")

    def is_sur_established(self):
        """This method checks if a SUR is established by looking at the number of consecutive goals without storing
            antitraces. 
            If this number of goals is higher than a threshold, the SUR is considered reliable enough (established).

        :return: 
        """
        n_goals_list_pos = []
        n_goals_list_neg = []
        for sensor in range(self.n_sensor):
            n_goals_list_pos.append(self.candidate_surs[sensor][0].number_of_goals_without_antitraces)
            n_goals_list_neg.append(self.candidate_surs[sensor][1].number_of_goals_without_antitraces)
        max_traces = max(max(n_goals_list_neg), max(n_goals_list_pos))
        if not self.established:
            if max_traces >= self.tb_max:
                self.established = True
                if max(n_goals_list_neg) > max(n_goals_list_pos):
                    i = n_goals_list_neg.index(max(n_goals_list_neg))
                    self.sur_established_type = 'neg'
                else:
                    i = n_goals_list_pos.index(max(n_goals_list_pos))
                    self.sur_established_type = 'pos'
                self.sur_established = i + 1
            else:
                self.established = False
