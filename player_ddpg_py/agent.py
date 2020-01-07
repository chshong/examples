# Author(s): Luiz Felipe Vecchietti, Chansol Hong, Inbae Jeong
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

from __future__ import print_function
import numpy as np
import sys
from ddpg import *
from converter import *

#reset_reason
NONE = 0
GAME_START = 1
SCORE_MYTEAM = 2
SCORE_OPPONENT = 3
GAME_END = 4
DEADLOCK = 5
GOALKICK = 6
CORNERKICK = 7
PENALTYKICK = 8
HALFTIME = 9
EPISODE_END = 10

#game_state
STATE_DEFAULT = 0
STATE_KICKOFF = 1
STATE_GOALKICK = 2
STATE_CORNERKICK = 3
STATE_PENALTYKICK = 4

#coordinates
MY_TEAM = 0
OP_TEAM = 1
BALL = 2
X = 0
Y = 1
TH = 2
ACTIVE = 3
TOUCH = 4



class Agent(object):
    def __init__(self, info):
        # Here you have the information of the game (virtual init() in random_walk.cpp)
        # List: game_time, number_of_robots
        #       field, goal, penalty_area, goal_area, resolution Dimension: [x, y]
        #       ball_radius, ball_mass,
        #       robot_size, robot_height, axle_length, robot_body_mass, ID: [0, 1, 2, 3, 4]
        #       wheel_radius, wheel_mass, ID: [0, 1, 2, 3, 4]
        #       max_linear_velocity, max_torque, codewords, ID: [0, 1, 2, 3, 4]
        # self.game_time = info['game_time']
        self.number_of_robots = info['number_of_robots']
        self.field = np.array(info['field'])
        self.goal = np.array(info['goal'])

        self.max_linear_velocity = info['max_linear_velocity']
        # self.max_torque = info['max_torque']
        # self.codewords = info['codewords']

        self.colorChannels = 3

        # DDPG parameters
        self.training = True
        self.load_model = False

        self.a_dim = 2
        self.a_bound = [self.field[X]/2 + self.goal[X]/2, self.field[Y]/2]
        self.converter = Converter(info, self.a_bound, 4)

        self.s_dim = self.converter.getStateDim()

        self.message_length = 4

        # Training setups
        self.episode = 0
        self.episode_step = 0

        self.ddpg_agent = DDPG(self.s_dim, self.a_dim, self.a_bound, self.number_of_robots, self.message_length, training=self.training)
        if self.load_model:
            self.ddpg_agent.load_model()
            self.ddpg_agent.init_update_target_network()
        self.loggers = self.ddpg_agent.loggers

    def update(self, frame):
        cur_state = self.converter.frame2state(frame)

        terminal = True if frame.reset_reason == SCORE_MYTEAM or frame.reset_reason == SCORE_OPPONENT or frame.reset_reason == EPISODE_END else False

        if self.episode_step > 0:
            rw = self.getReward(frame.game_state, frame.reset_reason)
            self.loggers.add_reward(rw)
            self.ddpg_agent.buffer_stack(self.prev_state, self.action, terminal, cur_state, rw)
            self.ddpg_agent.update()

        if terminal:
            self.loggers.write_summaries(self.episode_step, self.episode)
            self.episode += 1
            self.episode_step = 0

        self.action = self.ddpg_agent.predict(np.reshape(cur_state, (1, self.s_dim)))[0]
        wheels = self.converter.actions2speeds(self.action, frame.coordinates[MY_TEAM])
        # print(wheels)

        self.prev_state = cur_state
        self.episode_step += 1

        return wheels

    def getReward(self, game_state, reset_reason):
        if reset_reason == SCORE_MYTEAM:
            return [0.25, 0.5, 0.5, 1, 1]
        if reset_reason == SCORE_OPPONENT:
            return [-1, -0.5, -0.5, -0.25, -0.25]
        else:
            return [0, 0, 0, 0, 0]

    def __del__(self):
        print("I'm dying")
