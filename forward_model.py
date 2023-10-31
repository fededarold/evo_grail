import math

from scipy.spatial import distance


class ForwardModel(object):
    """
    Class tha represents a Forward Model, a model of the World.

    It contains the methods needed to predict the next State Space created with the application of a particular
    action in the actual State Space.
    """

    def predicted_state(self, action, scenario_data, w=75, w_act=20, h=60):
        """Return the predicted sensorization vector calculated as a function of the actual states and the particular
        actions applied to the robots.

        :param bax_l_action: Tuple containing the action applied (vel, angle) to the left arm of the Baxter robot
        :param w: width of the rectangle that represents the robot
        :param w_act: width of the rectangle that represents the robot actuator
        :param h: height of the rectangle that represents the robot and the rectangle that represents the robot actuator
        :return: sens:Tuple containing the distances between the ball and the robots' actuators and the reward
        """

        (bax_l_action) = action
        (bax_l_pos, bax_l_angle, obj1_pos, obj1_state, obj2_pos, obj2_state, obj3_pos, obj3_state, obj4_pos, obj4_state,
         obj5_pos, obj5_state, obj6_pos, obj6_state, obj7_pos, obj7_state, obj8_pos, obj8_state) = scenario_data

        # Predicted Baxter left arm position after applying the particular action
        bax_l_act_new_pos = self.get_act_new_pos(bax_l_pos, bax_l_angle, bax_l_action, w, h, w_act, h)
        sens = self.get_sensorization(bax_l_act_new_pos, obj1_pos, obj1_state, obj2_pos, obj2_state, obj3_pos,
                                      obj3_state, obj4_pos, obj4_state, obj5_pos, obj5_state, obj6_pos, obj6_state,
                                      obj7_pos, obj7_state, obj8_pos, obj8_state)

        return sens

    @staticmethod
    def get_ball_pos(ball_pos, situation, rob_pos, bax_l_pos):
        """Returns the predicted position of the ball according to its current situation.

        :param rob_pos: 
        :param ball_pos: Tuple containing the position (x, y) of the center of the ball
        :param situation: String that indicates the situation of the ball in the scenario ('robobo','baxter_rarm', 'bxter_larm' or else)
        :param bax_l_pos: Tuple containing the new position (x, y) of the center of the left arm of the Baxter robot
        :return: Tuple containing the new position (x, y) of the center of the ball
        """

        if situation == 'baxter_larm':
            ball_new_pos = bax_l_pos
        elif situation == 'robobo':
            ball_new_pos = rob_pos
        else:
            ball_new_pos = ball_pos

        return ball_new_pos

    def get_sensorization(self, bax_l_pos, obj1_pos, obj1_state, obj2_pos, obj2_state, obj3_pos, obj3_state, obj4_pos,
                          obj4_state, obj5_pos, obj5_state, obj6_pos, obj6_state, obj7_pos, obj7_state, obj8_pos,
                          obj8_state):
        """Return a sensorization vector with the distances between the ball and the robots' actuators.

        :param button_color: 
        :param obstacle_color: 
        :param ball_color: 
        :param button_pos: 
        :param obstacle_pos: 
        :param bax_l_pos: 
        :param ball_pos: Tuple containing the new position (x, y) of the center of the ball
        :return: Tuple containing the distances between the ball and the robots' actuators
        """

        max_dist = 2420.0
        d1 = distance.euclidean(bax_l_pos, obj1_pos)
        d1_norm = self.normalize_value(d1, max_dist)
        d2 = distance.euclidean(bax_l_pos, obj2_pos)
        d2_norm = self.normalize_value(d2, max_dist)
        d3 = distance.euclidean(bax_l_pos, obj3_pos)
        d3_norm = self.normalize_value(d3, max_dist)
        d4 = distance.euclidean(bax_l_pos, obj4_pos)
        d4_norm = self.normalize_value(d4, max_dist)
        d5 = distance.euclidean(bax_l_pos, obj5_pos)
        d5_norm = self.normalize_value(d5, max_dist)
        d6 = distance.euclidean(bax_l_pos, obj6_pos)
        d6_norm = self.normalize_value(d6, max_dist)
        d7 = distance.euclidean(bax_l_pos, obj7_pos)
        d7_norm = self.normalize_value(d7, max_dist)
        d8 = distance.euclidean(bax_l_pos, obj8_pos)
        d8_norm = self.normalize_value(d8, max_dist)
        min_dist_robot = 50
        if not obj1_state:
            if d1 < min_dist_robot:
                obj1_state = True
        if not obj2_state:
            if d2 < min_dist_robot:
                obj2_state = True
        if not obj3_state:
            if d3 < min_dist_robot:
                obj3_state = True
        if not obj4_state:
            if d4 < min_dist_robot:
                obj4_state = True
        if not obj5_state:
            if d5 < min_dist_robot:
                obj5_state = True
        if not obj6_state:
            if d6 < min_dist_robot:
                obj6_state = True
        if not obj7_state:
            if d7 < min_dist_robot:
                obj7_state = True
        if not obj8_state:
            if d8 < min_dist_robot:
                obj8_state = True
        return d1_norm, d2_norm, d3_norm, d4_norm, d5_norm, d6_norm

    @staticmethod
    def normalize_value(value, max_value, min_value=0.0):
        return (value - min_value) / (max_value - min_value)

    def get_act_new_pos(self, x_y, angle, rel_angle, w, h, w_act, h_act, vel=50):
        """Returns the new position of the actuator of the robot.

        :param angle: actual angle of the robot
        :param rel_angle: candidate relative angle to apply to the robot
        :param w:  width of the rectangle that represents the robot
        :param h: height of the rectangle that represents the robot 
        :param w_act:  width of the rectangle that represents the robot actuator
        :param h_act: height of the rectangle that represents the robot actuator
        :param vel: velocity of movement (default = 10)
        :return: Tuple containing the new position (x, y) of the center of the robot actuator
        """
        (x, y) = x_y

        angle += rel_angle

        x_new, y_new = self.get_new_pos((x, y), vel, angle, w, h)

        lim_baxter_x = (100, 2400)
        lim_baxter_y = (50, 800)

        if (x_new > lim_baxter_x[0]) and (x_new < lim_baxter_x[1]) and (y_new > lim_baxter_y[0]) and (
                y_new < lim_baxter_y[1]):
            x_act, y_act = (x_new + w * math.cos(angle * math.pi / 180), y_new + w * math.sin(angle * math.pi / 180))

            xc_act = x_act + w_act / 2 * math.cos(angle * math.pi / 180) - h_act / 2 * math.sin(angle * math.pi / 180)
            yc_act = y_act + w_act / 2 * math.sin(angle * math.pi / 180) + h_act / 2 * math.cos(angle * math.pi / 180)
            return xc_act, yc_act
        else:
            return x, y

    @staticmethod
    def get_new_pos(x_y, vel, angle, w, h):
        """Returns the new position of the robot after applying a particular action defined by (vel, angle).

        :param vel: A value that represents the velocity of motion
        :param angle: A value that represents the orientation choosen to move forward (0-360)
        :param w:  width of the rectangle that represents the robot
        :param h: height of the rectangle that represents the robot
        :return: A tuple containing the position (x, y) of the left lower vertex of the robot after applying a particular action
        """
        (x, y) = x_y
        # New center position
        new_xc2 = x + vel * math.cos(angle * math.pi / 180)
        new_yc2 = y + vel * math.sin(angle * math.pi / 180)
        # New x, y position
        new_x2 = new_xc2 - w / 2 * math.cos(angle * math.pi / 180) + h / 2 * math.sin(angle * math.pi / 180)
        new_y2 = new_yc2 - w / 2 * math.sin(angle * math.pi / 180) - h / 2 * math.cos(angle * math.pi / 180)

        return new_x2, new_y2
