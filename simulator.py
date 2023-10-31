import math

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.spatial import distance


class Sim(object):
    """ Class that implements a Simulator
    
    This simulator makes possible to make/test different experiments in a virtual scenario.
    It contains the two arms of the Baxter robot, the Robobo! robot, some boxes and a
    ball.
    
    The implemented methods allow the user to move both robots throw the scenario (with
    and without the ball), get distances and relative angles between the different objects 
    and get/set the position of all of them.
    """

    def __init__(self):
        """Create the different objects present in the Simulator and place them in it"""

        # Enable simulation visualization (see the objects moving)
        self.visualize = False

        # self.ball_1 = plt.Circle((300, 750), 40, fc='red', alpha=0.25, label='ball_1')
        self.ball_1 = plt.Circle((300, 650), 40, fc='red', alpha=0.25, label='ball_1')
        self.ball_2 = plt.Circle((700, 650), 40, fc='green', alpha=0.25, label='ball_2')
        self.ball_3 = plt.Circle((1100, 700), 40, fc='blue', alpha=0.25, label='ball_3')
        self.ball_4 = plt.Circle((1500, 700), 40, fc='orange', alpha=0.25, label='ball_4')
        self.ball_5 = plt.Circle((1900, 650), 40, fc='purple', alpha=0.25, label='ball_5')
        self.ball_6 = plt.Circle((2300, 600), 40, fc='cyan', alpha=0.25, label='ball_6')
        self.ball_7 = plt.Circle((280, 250), 40, fc='gold', alpha=0.25, label='ball_7')
        self.ball_8 = plt.Circle((2300, 250), 40, fc='grey', alpha=0.25, label='ball_8')

        self.ball_active_goal = plt.Circle((500, 900), 40, fc='white', alpha=1.0, label='ball_active_goal')
        self.ball_active_subgoal = plt.Circle((1200, 900), 40, fc='white', alpha=1.0, label='ball_active_subgoal')
        self.ball_active_context = plt.Circle((2000, 900), 40, fc='white', alpha=1.0, label='ball_context')

        # Robots
        self.robobo = patches.Rectangle((700, 300), 75, 60, angle=0.0, fc='cyan', label='robobo')
        self.robobo_act = patches.Rectangle((775, 300), 20, 60, angle=0.0, fc='blue', label='robobo_actuator')
        self.baxter_rarm = patches.Rectangle((2000, 50), 75, 60, angle=0.0, fc=(0.8, 0, 0.2), label='baxter_rarm')
        self.baxter_rarm_act = patches.Rectangle((2075, 50), 20, 60, angle=0.0, fc='black', label='baxter_rarm_act')
        self.baxter_larm = patches.Rectangle((1600, 50), 75, 60, angle=0.0, fc=(0.8, 0, 0.2), label='baxter_larm')
        self.baxter_larm_act = patches.Rectangle((1675, 50), 20, 60, angle=0.0, fc='black', label='baxter_larm_act')

        self.baxter_figure = patches.Circle((2700, 264), 12, fc=(0.8, 0, 0, 1))
        self.baxter_figure_1 = patches.Circle((2700, 264), 12, fc=(0.8, 0, 0, 0.8))
        self.baxter_figure_2 = patches.Circle((2700, 264), 12, fc=(0.8, 0, 0, 0.6))
        self.baxter_figure_3 = patches.Circle((2700, 264), 12, fc=(0.8, 0, 0, 0.4))
        self.baxter_figure_4 = patches.Circle((2700, 264), 12, fc=(0.8, 0, 0, 0.2))
        self.baxter_figure_5 = patches.Circle((2700, 264), 12, fc=(0.8, 0, 0, 0.0))

        self.fig = plt.figure()

        '''FEDE '''
        #self.fig.canvas.set_window_title('Simulator')
        # self.fig.canvas.setWindowTitle('Simulator')
        ''' FEDE '''

        self.ax = plt.axes(xlim=(0, 3500), ylim=(0, 1000))
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)

        xy1 = (
            self.ball_active_context.center[0],
            self.ball_active_context.center[1] + 1.2 * self.ball_active_context.radius)
        self.ax.annotate("CONTEXT", xy=xy1, fontsize=10, ha="center")
        xy2 = (self.ball_active_goal.center[0], self.ball_active_goal.center[1] + 1.2 * self.ball_active_goal.radius)
        self.ax.annotate("GOAL", xy=xy2, fontsize=10, ha="center")
        xy3 = (
            self.ball_active_subgoal.center[0],
            self.ball_active_subgoal.center[1] + 1.2 * self.ball_active_subgoal.radius)
        self.ax.annotate("SUB-GOAL", xy=xy3, fontsize=10, ha="center")

        # Movement boundaries
        plt.axhline(y=800, xmin=0.0286, xmax=0.686, linestyle='--', color='grey')
        plt.axhline(y=50, xmin=0.0286, xmax=0.686, linestyle='--', color='grey')
        plt.axvline(x=100, ymin=0.05, ymax=0.80, linestyle='--', color='grey')
        plt.axvline(x=2400, ymin=0.05, ymax=0.80, linestyle='--', color='grey')
        self.ball_position = None  # Indicates where is the ball: robobo, baxter_larm, baxter_rarm, box1, box2 or None

        # Efects
        self.ball_1_active = False
        self.ball_2_active = False
        self.ball_3_active = False
        self.ball_4_active = False
        self.ball_5_active = False
        self.ball_6_active = False
        self.ball_7_active = False
        self.ball_8_active = False

        # Active goal
        self.active_goal = None
        self.active_subgoal = None
        self.active_context = None

        # Show figure and patches
        ## Objects
        self.fig.show()
        self.ax.add_patch(self.ball_1)
        self.ax.add_patch(self.ball_2)
        self.ax.add_patch(self.ball_3)
        self.ax.add_patch(self.ball_4)
        self.ax.add_patch(self.ball_5)
        self.ax.add_patch(self.ball_6)
        self.ax.add_patch(self.ball_7)
        self.ax.add_patch(self.ball_8)
        self.ax.add_patch(self.ball_active_goal)
        self.ax.add_patch(self.ball_active_subgoal)
        self.ax.add_patch(self.ball_active_context)

        ## Robots
        self.ax.add_patch(self.baxter_larm)
        self.ax.add_patch(self.baxter_larm_act)

        # State space test
        plt.axhline(y=950, xmin=0.771, xmax=0.967, linestyle='-', color='black', linewidth=1.3)
        plt.axhline(y=264, xmin=0.771, xmax=0.967, linestyle='-', color='black', linewidth=1.3)
        plt.axhline(y=364, xmin=0.771, xmax=0.967, linestyle='--', color='grey')
        plt.axhline(y=464, xmin=0.771, xmax=0.967, linestyle='--', color='grey')
        plt.axhline(y=564, xmin=0.771, xmax=0.967, linestyle='--', color='grey')
        plt.axhline(y=664, xmin=0.771, xmax=0.967, linestyle='--', color='grey')
        plt.axhline(y=764, xmin=0.771, xmax=0.967, linestyle='--', color='grey')
        plt.axhline(y=864, xmin=0.771, xmax=0.967, linestyle='--', color='grey')
        plt.axvline(x=2700, ymin=0.264, ymax=0.950, linestyle='-', color='black', linewidth=1.3)
        plt.axvline(x=3386, ymin=0.264, ymax=0.950, linestyle='-', color='black', linewidth=1.3)
        plt.axvline(x=2800, ymin=0.264, ymax=0.950, linestyle='--', color='grey')
        plt.axvline(x=2900, ymin=0.264, ymax=0.950, linestyle='--', color='grey')
        plt.axvline(x=3000, ymin=0.264, ymax=0.950, linestyle='--', color='grey')
        plt.axvline(x=3100, ymin=0.264, ymax=0.950, linestyle='--', color='grey')
        plt.axvline(x=3200, ymin=0.264, ymax=0.950, linestyle='--', color='grey')
        plt.axvline(x=3300, ymin=0.264, ymax=0.950, linestyle='--', color='grey')
        plt.axvline(x=2500)
        self.ax.add_patch(self.baxter_figure)
        self.ax.add_patch(self.baxter_figure_1)
        self.ax.add_patch(self.baxter_figure_2)
        self.ax.add_patch(self.baxter_figure_3)
        self.ax.add_patch(self.baxter_figure_4)
        self.ax.add_patch(self.baxter_figure_5)

        # self.restart_scenario()

        self.secuencia = 1

    def ball_1_set_pos(self, x_y):
        """Set the ball center position"""
        self.ball_1.center = x_y

    def ball_2_set_pos(self, x_y):
        """Set the ball center position"""
        self.ball_2.center = x_y

    def ball_3_set_pos(self, x_y):
        """Set the ball center position"""
        self.ball_3.center = x_y

    def ball_4_set_pos(self, x_y):
        """Set the ball center position"""
        self.ball_4.center = x_y

    def ball_5_set_pos(self, x_y):
        """Set the ball center position"""
        self.ball_5.center = x_y

    def ball_6_set_pos(self, x_y):
        """Set the ball center position"""
        self.ball_6.center = x_y

    def ball_7_set_pos(self, x_y):
        """Set the ball center position"""
        self.ball_7.center = x_y

    def ball_8_set_pos(self, x_y):
        """Set the ball center position"""
        self.ball_8.center = x_y

    def baxter_larm_set_pos(self, x_y):
        """Set the Baxter's left arm center position (and its actuator) checking if it is inside its movement limits"""
        (x, y) = x_y
        w = self.baxter_larm.get_width()
        h = self.baxter_larm.get_height()
        # New x, y position
        new_x = x - w / 2 * math.cos(self.baxter_larm.angle * math.pi / 180) + h / 2 * math.sin(
            self.baxter_larm.angle * math.pi / 180)
        new_y = y - w / 2 * math.sin(self.baxter_larm.angle * math.pi / 180) - h / 2 * math.cos(
            self.baxter_larm.angle * math.pi / 180)
        if self.check_limits((x, y), 'baxter'):
            self.baxter_larm.xy = (new_x, new_y)
            self.baxter_larm_act.xy = (
                new_x + w * math.cos(self.baxter_larm_act.angle * math.pi / 180),
                new_y + w * math.sin(self.baxter_larm_act.angle * math.pi / 180)
            )

    def baxter_larm_set_angle(self, angle):
        """Set the Baxter's left arm angle (and its actuator angle and adjust its position)"""
        centre = self.baxter_larm_get_pos()
        self.baxter_larm.angle = angle
        self.baxter_larm_act.angle = angle
        self.baxter_larm_set_pos(centre)
        w = self.baxter_larm.get_width()
        self.baxter_larm_act.xy = (self.baxter_larm.get_x() + w * math.cos(self.baxter_larm_act.angle * math.pi / 180),
                                   self.baxter_larm.get_y() + w * math.sin(self.baxter_larm_act.angle * math.pi / 180))
        if self.ball_position == 'baxter_larm':
            self.ball_set_pos(self.baxter_larm_act_get_pos())

    def ball_1_get_pos(self):
        """Return the position of the center of the ball"""
        return self.ball_1.center

    def ball_2_get_pos(self):
        """Return the position of the center of the ball"""
        return self.ball_2.center

    def ball_3_get_pos(self):
        """Return the position of the center of the ball"""
        return self.ball_3.center

    def ball_4_get_pos(self):
        """Return the position of the center of the ball"""
        return self.ball_4.center

    def ball_5_get_pos(self):
        """Return the position of the center of the ball"""
        return self.ball_5.center

    def ball_6_get_pos(self):
        """Return the position of the center of the ball"""
        return self.ball_6.center

    def ball_7_get_pos(self):
        """Return the position of the center of the ball"""
        return self.ball_7.center

    def ball_8_get_pos(self):
        """Return the position of the center of the ball"""
        return self.ball_8.center

    def baxter_larm_get_pos(self):
        """Return the position of the center of the Baxter's left arm"""
        w = self.baxter_larm.get_width()
        h = self.baxter_larm.get_height()
        x, y = self.baxter_larm.xy
        x_c = x + w / 2 * math.cos(self.baxter_larm.angle * math.pi / 180) - h / 2 * math.sin(
            self.baxter_larm.angle * math.pi / 180)
        y_c = y + w / 2 * math.sin(self.baxter_larm.angle * math.pi / 180) + h / 2 * math.cos(
            self.baxter_larm.angle * math.pi / 180)
        return x_c, y_c

    def baxter_larm_act_get_pos(self):
        """Return the position of the center of the Baxter's left arm actuator"""
        w = self.baxter_larm_act.get_width()
        h = self.baxter_larm_act.get_height()
        x, y = self.baxter_larm_act.xy
        x_c = x + w / 2 * math.cos(self.baxter_larm_act.angle * math.pi / 180) - h / 2 * math.sin(
            self.baxter_larm_act.angle * math.pi / 180)
        y_c = y + w / 2 * math.sin(self.baxter_larm_act.angle * math.pi / 180) + h / 2 * math.cos(
            self.baxter_larm_act.angle * math.pi / 180)
        return x_c, y_c

    def baxter_larm_get_angle(self):
        """Return the angle of the left arm of the Baxter robot"""
        return self.baxter_larm.angle

    @staticmethod
    def normalize_value(value, max_value, min_value=0.0):
        return (value - min_value) / (max_value - min_value)

    @staticmethod
    def get_relative_angle(x1_y1, x2_y2):
        """Return the relative angle between two points"""
        (x1, y1) = x1_y1
        (x2, y2) = x2_y2
        return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

    def move_baxter_larm(self, vel=40):
        """Move Baxter's left arm wih a specific velocity (default 10)"""
        x, y = self.baxter_larm_get_pos()
        # print "X, y move arm ", (x, y)
        self.baxter_larm_set_pos((x + vel * math.cos(self.baxter_larm.angle * math.pi / 180),
                                  y + vel * math.sin(self.baxter_larm.angle * math.pi / 180)))
        if self.ball_position == 'baxter_larm':
            self.ball_set_pos(self.baxter_larm_act_get_pos())

    def baxter_larm_action(self, relative_angle, vel=40):
        """Move baxter left arm with a specific angle and velocity (default 10)"""
        angle = self.baxter_larm.angle + relative_angle
        self.baxter_larm_set_angle(angle)
        self.move_baxter_larm(vel)
        self.world_rules()
        self.baxter_state_space()

    def apply_action(self, relative_angles, vel_rob=25, vel_baxt=50):
        """Move robobo and baxter left arm with a specific angle and velocity (default 20)"""
        self.baxter_larm_action(relative_angles, vel_baxt)
        self.baxter_state_space()
        if self.visualize:
            self.fig.canvas.draw()
            plt.pause(0.01)

    def get_reward(self):
        """Return the reward checking if the ball is inside one of the boxes"""
        if (self.ball_1_active and self.active_goal == 'ball_1') or (
                self.ball_2_active and self.active_goal == 'ball_2') or (
                self.ball_3_active and self.active_goal == 'ball_3') or (
                self.ball_4_active and self.active_goal == 'ball_4') or (
                self.ball_5_active and self.active_goal == 'ball_5') or (
                self.ball_6_active and self.active_goal == 'ball_6') or (
                self.ball_7_active and self.active_goal == 'ball_7') or (
                self.ball_8_active and self.active_goal == 'ball_8'):
            reward = True
        else:
            reward = False
        return reward

    def get_subgoal_reward(self):
        """Return the reward checking if the ball is inside one of the boxes"""
        if (self.ball_1_active and self.active_subgoal == 'ball_1') or (
                self.ball_2_active and self.active_subgoal == 'ball_2') or (
                self.ball_3_active and self.active_subgoal == 'ball_3') or (
                self.ball_4_active and self.active_subgoal == 'ball_4') or (
                self.ball_5_active and self.active_subgoal == 'ball_5') or (
                self.ball_6_active and self.active_subgoal == 'ball_6') or (
                self.ball_7_active and self.active_goal == 'ball_7') or (
                self.ball_8_active and self.active_goal == 'ball_8'):
            reward = True
        else:
            reward = False
        return reward

    def get_virtual_reward(self):
        """Return the virtual reward according to the interest. When an effect occurs"""
        if self.button_pushed or self.ball_moved:
            return True
        else:
            return False

    def get_sensorization(self):
        """Return a sensorization vector with the distance between the object in which the robot is focused and its
         actuator, the color of this object..."""

        max_dist = 2420.0
        d1 = self.normalize_value(distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_1_get_pos()), max_dist)
        d2 = self.normalize_value(distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_2_get_pos()), max_dist)
        d3 = self.normalize_value(distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_3_get_pos()), max_dist)
        d4 = self.normalize_value(distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_4_get_pos()), max_dist)
        d5 = self.normalize_value(distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_5_get_pos()), max_dist)
        d6 = self.normalize_value(distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_6_get_pos()), max_dist)
        d7 = self.normalize_value(distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_7_get_pos()), max_dist)
        d8 = self.normalize_value(distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_8_get_pos()), max_dist)

        return d1, d2, d3, d4, d5, d6

    def get_goals_state(self):
        return self.ball_1_active, self.ball_2_active, self.ball_3_active, self.ball_4_active, self.ball_5_active, self.ball_6_active  # , self.ball_7_active#, self.ball_8_active

    def get_scenario_data(self):
        """Scenario data needed to predict future states using the world model"""
        data = []
        data.append(self.baxter_larm_act_get_pos())
        data.append(self.baxter_larm_get_angle())
        data.append(self.ball_1_get_pos())
        data.append(self.ball_1_active)
        data.append(self.ball_2_get_pos())
        data.append(self.ball_2_active)
        data.append(self.ball_3_get_pos())
        data.append(self.ball_3_active)
        data.append(self.ball_4_get_pos())
        data.append(self.ball_4_active)
        data.append(self.ball_5_get_pos())
        data.append(self.ball_5_active)
        data.append(self.ball_6_get_pos())
        data.append(self.ball_6_active)
        data.append(self.ball_7_get_pos())
        data.append(self.ball_7_active)
        data.append(self.ball_8_get_pos())
        data.append(self.ball_8_active)

        return tuple(data)

    def world_rules(self):
        """Establish the ball position in the scenario"""
        # Set minimum distances for the ball to be inside the box or grasped by a robot
        min_dist_box = 80
        min_dist_robot = 50
        # Ball position: Check where the ball has to be (check distances)
        d1 = distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_1_get_pos())
        d2 = distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_2_get_pos())
        d3 = distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_3_get_pos())
        d4 = distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_4_get_pos())
        d5 = distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_5_get_pos())
        d6 = distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_6_get_pos())
        d7 = distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_7_get_pos())
        d8 = distance.euclidean(self.baxter_larm_act_get_pos(), self.ball_8_get_pos())

        # Scenario 1
        if self.secuencia == 1:
            if d1 < min_dist_robot:  # and not self.ball_2_active and not self.ball_3_active:# and self.active_context == 'c0':
                self.ball_1_active = True
                self.ball_1.set_alpha(1.0)
            elif d2 < min_dist_robot:  # and not self.ball_1_active:# and not self.ball_3_active:# and self.active_context == 'c1':# and self.ball_1_active:
                self.ball_2_active = True
                self.ball_2.set_alpha(1.0)
            elif d3 < min_dist_robot:  # and not self.ball_1_active: # and self.active_context == 'c1':
                self.ball_3_active = True
                self.ball_3.set_alpha(1.0)
            elif d4 < min_dist_robot and self.ball_2_active:
                self.ball_4_active = True
                self.ball_4.set_alpha(1.0)
            elif d5 < min_dist_robot and self.ball_3_active:
                self.ball_5_active = True
                self.ball_5.set_alpha(1.0)
            elif d6 < min_dist_robot and self.ball_5_active:  # and self.ball_4_active:
                self.ball_6_active = True
                self.ball_6.set_alpha(1.0)
            # elif d7 < min_dist_robot and self.ball_5_active:  # and self.ball_1_active:
            #     self.ball_7_active = True
            #     self.ball_7.set_alpha(1.0)
            # if self.ball_6_active and self.ball_7_active:
            #     self.ball_8_active = True
            #     self.ball_8.set_alpha(1.0)
        elif self.secuencia == 2:
            if d1 < min_dist_robot:  # and not self.ball_2_active and not self.ball_3_active:# and self.active_context == 'c0':
                self.ball_1_active = True
                self.ball_1.set_alpha(1.0)
            elif d2 < min_dist_robot:  # and not self.ball_1_active:# and not self.ball_3_active:# and self.active_context == 'c1':# and self.ball_1_active:
                self.ball_2_active = True
                self.ball_2.set_alpha(1.0)
            elif d3 < min_dist_robot:  # and not self.ball_1_active: # and self.active_context == 'c1':
                self.ball_3_active = True
                self.ball_3.set_alpha(1.0)
            elif d4 < min_dist_robot and not self.ball_2_active:
                self.ball_4_active = True
                self.ball_4.set_alpha(1.0)
            elif d5 < min_dist_robot and not self.ball_3_active:
                self.ball_5_active = True
                self.ball_5.set_alpha(1.0)
            elif d6 < min_dist_robot and self.ball_5_active and self.ball_1_active:  # and self.ball_3_active:  # and self.ball_1_active:
                self.ball_6_active = True
                self.ball_6.set_alpha(1.0)
            # elif d7 < min_dist_robot and self.ball_5_active and self.ball_3_active:  # and self.ball_1_active:
            #     self.ball_7_active = True
            #     self.ball_7.set_alpha(1.0)
            # if self.ball_3_active and self.ball_6_active and self.ball_7_active:
            #     self.ball_8_active = True
            #     self.ball_8.set_alpha(1.0)
        else:
            if d1 < min_dist_robot:  # and not self.ball_2_active and not self.ball_3_active:# and self.active_context == 'c0':
                self.ball_1_active = True
                self.ball_1.set_alpha(1.0)
            elif d2 < min_dist_robot:  # and not self.ball_1_active:# and not self.ball_3_active:# and self.active_context == 'c1':# and self.ball_1_active:
                self.ball_2_active = True
                self.ball_2.set_alpha(1.0)
            elif d3 < min_dist_robot:  # and not self.ball_1_active: # and self.active_context == 'c1':
                self.ball_3_active = True
                self.ball_3.set_alpha(1.0)
            elif d4 < min_dist_robot and self.ball_2_active:
                self.ball_4_active = True
                self.ball_4.set_alpha(1.0)
            elif d5 < min_dist_robot and self.ball_1_active:
                self.ball_5_active = True
                self.ball_5.set_alpha(1.0)
            elif d6 < min_dist_robot and self.ball_4_active and self.ball_5_active and not self.ball_3_active:  # and self.ball_1_active:
                self.ball_6_active = True
                self.ball_6.set_alpha(1.0)
            # elif d7 < min_dist_robot and self.ball_5_active and not self.ball_3_active:  # and self.ball_1_active:
            #     self.ball_7_active = True
            #     self.ball_7.set_alpha(1.0)
            # if not self.ball_3_active and self.ball_6_active and self.ball_7_active:
            #     self.ball_8_active = True
            #     self.ball_8.set_alpha(1.0)

    def choose_active_goal(self, iterations):
        """Choose the active goal in each instant of time"""
        if iterations < 2000:
            self.set_active_goal('ball_1')
        elif iterations < 5000:
            self.set_active_goal('ball_2')
        elif iterations < 9000:
            self.set_active_goal('ball_3')
        elif iterations < 9300:
            self.set_active_goal('ball_2')
        elif iterations < 9600:
            self.set_active_goal('ball_1')
        else:
            self.set_active_goal('ball_3')

    def set_active_goal(self, goal):
        self.active_goal = goal
        self.show_active_goal()

    def show_active_goal(self):
        if self.active_goal == 'ball_1':
            self.ball_active_goal.set_facecolor(self.ball_1.get_facecolor())
        elif self.active_goal == 'ball_2':
            self.ball_active_goal.set_facecolor(self.ball_2.get_facecolor())
        elif self.active_goal == 'ball_3':
            self.ball_active_goal.set_facecolor(self.ball_3.get_facecolor())
        elif self.active_goal == 'ball_4':
            self.ball_active_goal.set_facecolor(self.ball_4.get_facecolor())
        elif self.active_goal == 'ball_5':
            self.ball_active_goal.set_facecolor(self.ball_5.get_facecolor())
        elif self.active_goal == 'ball_6':
            self.ball_active_goal.set_facecolor(self.ball_6.get_facecolor())
        elif self.active_goal == 'ball_7':
            self.ball_active_goal.set_facecolor(self.ball_7.get_facecolor())
        elif self.active_goal == 'ball_8':
            self.ball_active_goal.set_facecolor(self.ball_8.get_facecolor())

    def set_active_subgoal(self, goal):
        self.active_subgoal = goal
        self.show_active_subgoal()

    def show_active_subgoal(self):
        if self.active_subgoal == 'ball_1':
            self.ball_active_subgoal.set_facecolor(self.ball_1.get_facecolor())
        elif self.active_subgoal == 'ball_2':
            self.ball_active_subgoal.set_facecolor(self.ball_2.get_facecolor())
        elif self.active_subgoal == 'ball_3':
            self.ball_active_subgoal.set_facecolor(self.ball_3.get_facecolor())
        elif self.active_subgoal == 'ball_4':
            self.ball_active_subgoal.set_facecolor(self.ball_4.get_facecolor())
        elif self.active_subgoal == 'ball_5':
            self.ball_active_subgoal.set_facecolor(self.ball_5.get_facecolor())
        elif self.active_subgoal == 'ball_6':
            self.ball_active_subgoal.set_facecolor(self.ball_6.get_facecolor())
        elif self.active_subgoal == 'ball_7':
            self.ball_active_subgoal.set_facecolor(self.ball_7.get_facecolor())
        elif self.active_subgoal == 'ball_8':
            self.ball_active_subgoal.set_facecolor(self.ball_8.get_facecolor())

    def choose_active_context(self, iterations):
        """Choose the active goal in each instant of time"""
        self.set_active_context('c0')

    def set_active_context(self, context):
        self.active_context = context
        self.show_active_context()

    def show_active_context(self):
        if self.active_context == 'c0':
            self.ball_active_context.set_facecolor('black')
        elif self.active_context == 'c1':
            self.ball_active_context.set_facecolor('orange')

    @staticmethod
    def check_limits(x_y, robot_type):
        """Check if the next position of one of the robots is inside its movement limits"""
        (x, y) = x_y
        # Set limits for robots movements
        lim_robobo_x = (100, 1250)  # (x_min,x_max)
        lim_robobo_y = (50, 950)
        lim_baxter_x = (100, 2400)
        lim_baxter_y = (50, 800)
        # Movement boundaries
        result = 1
        if robot_type == 'robobo':
            if x < lim_robobo_x[0] or x > lim_robobo_x[1] or y < lim_robobo_y[0] or y > lim_robobo_y[1]:
                result = 0  # Do not move: it may crash
        elif robot_type == 'baxter':
            if x < lim_baxter_x[0] or x > lim_baxter_x[1] or y < lim_baxter_y[0] or y > lim_baxter_y[1]:
                result = 0  # Do not move: exceed movement boundaries

        return result

    def restart_scenario(self):
        self.baxter_larm_set_pos((np.random.randint(100, 2400), np.random.randint(50, 300)))

        self.ball_1.set_alpha(0.25)
        self.ball_2.set_alpha(0.25)
        self.ball_3.set_alpha(0.25)
        self.ball_4.set_alpha(0.25)
        self.ball_5.set_alpha(0.25)
        self.ball_6.set_alpha(0.25)
        self.ball_7.set_alpha(0.25)
        self.ball_8.set_alpha(0.25)
        self.ball_1_active = False
        self.ball_2_active = False
        self.ball_3_active = False
        self.ball_4_active = False
        self.ball_5_active = False
        self.ball_6_active = False
        self.ball_7_active = False
        self.ball_8_active = False
        self.world_rules()

    def restart_scenario_partially(self):
        self.baxter_larm_set_pos((np.random.randint(100, 2400), np.random.randint(50, 200)))
        self.world_rules()

    def baxter_state_space(self):
        x5, y5 = self.baxter_figure_4.center
        x4, y4 = self.baxter_figure_3.center
        x3, y3 = self.baxter_figure_2.center
        x2, y2 = self.baxter_figure_1.center
        x1, y1 = self.baxter_figure.center

        x, y = 2700 + self.get_sensorization()[0] / 4, 264 + self.get_sensorization()[1] / 4

        self.baxter_figure.center = (x, y)
        self.baxter_figure_1.center = (x1, y1)
        self.baxter_figure_2.center = (x2, y2)
        self.baxter_figure_3.center = (x3, y3)
        self.baxter_figure_4.center = (x4, y4)
        self.baxter_figure_5.center = (x5, y5)


if __name__ == '__main__':
    a = Sim()
