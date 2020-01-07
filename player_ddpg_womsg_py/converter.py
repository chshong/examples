from __future__ import division
import math
import numpy as np

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

class Converter(object):
    def __init__(self, info, bound, nstep=4):
        self.scale = 1.4
        self.mult_lin = 5.0
        self.mult_ang = 0.4
        self.damping = 0.35
        self.info = info
        self.bound = bound
        self.x_buffer = np.zeros((11, nstep))
        self.y_buffer = np.zeros((11, nstep))
        self.th_buffer = np.zeros((10, nstep))

    def frame2state(self, frame):
        x = []
        y = []
        th = []
        active = []
        for item in frame.coordinates[MY_TEAM]:
            x.append(item[X])
            y.append(item[Y])
            th.append(item[TH])
            active.append(item[ACTIVE])

        for item in frame.coordinates[OP_TEAM]:
            x.append(item[X])
            y.append(item[Y])
            th.append(item[TH])
            active.append(item[ACTIVE])

        x.append(frame.coordinates[BALL][X])
        y.append(frame.coordinates[BALL][Y])

        x = np.array(x)/self.bound[X]
        y = np.array(y)/self.bound[Y]
        th = np.array(th)/math.pi
        active = 2*np.array(active) - 1

        self.pushData(x, self.x_buffer, frame.reset_reason!=NONE)
        self.pushData(y, self.y_buffer, frame.reset_reason!=NONE)
        self.pushData(th, self.th_buffer, frame.reset_reason!=NONE)

        diff = np.concatenate((self.getDiff(self.x_buffer), self.getDiff(self.y_buffer), self.getDiff(self.th_buffer)))

        return np.append([frame.game_state/4., frame.reset_reason/10.], np.concatenate((x,y,th,active, diff)))

    def getStateDim(self):
        return 2*(self.x_buffer.shape[0] + self.y_buffer.shape[0] + self.th_buffer.shape[0]) + 10 + 2

    def pushData(self, data, buffer, reset):
        if reset:
            # fill up the records
            for i in range(buffer.shape[1]):
              buffer[:,:-1] = buffer[:,1:]; buffer[:,-1] = data
        else:
            buffer[:,:-1] = buffer[:,1:]; buffer[:,-1] = data

    def getDiff(self, buffer):
        return (buffer[:,-1] - buffer[:,0])/2

    def actions2speeds(self, actions, my_coordinates):
        speeds = np.array([])
        for i in range(5):
            robot_action = actions[2*i:2*i+2]
            robot_pos = my_coordinates[i][0:3]
            speeds = np.append(speeds, self.action2speed(robot_action, robot_pos))

        return list(speeds)

    def action2speed(self, target, cur_pos):
        ka = 0
        sign = 1

        # calculate how far the target position is from the robot
        diff = np.array(target) - cur_pos[:2]
        d_e = math.sqrt(np.sum(np.power(diff,2)))

        # calculate how much the direction is off
        desired_th = (math.pi / 2) if (list(diff) == [0, 0]) else math.atan2(diff[1], diff[0])
        d_th = self.adjustAngleRange(desired_th - cur_pos[2])

        # based on how far the target position is, set a parameter that
        # decides how much importance should be put into changing directions
        # farther the target is, less need to change directions fastly
        if (d_e > 1):
            ka = 17 / 90
        elif (d_e > 0.5):
            ka = 19 / 90
        elif (d_e > 0.3):
            ka = 21 / 90
        elif (d_e > 0.2):
            ka = 23 / 90
        else:
            ka = 25 / 90

        # if the target position is at rear of the robot, drive backward instead
        if (d_th > math.radians(95)):
            d_th -= math.pi
            sign = -1
        elif (d_th < math.radians(-95)):
            d_th += math.pi
            sign = -1

        # if the direction is off by more than 85 degrees,
        # make a turn first instead of start moving toward the target
        if (abs(d_th) > math.radians(85)):
            return [-self.mult_ang*d_th, self.mult_ang*d_th]
        # otherwise
        else:
            # scale the angular velocity further down if the direction is off by less than 40 degrees
            if (d_e < 5 and abs(d_th) < math.radians(40)):
                ka = 0.1
            ka *= 4

            return [sign * self.scale * (self.mult_lin * (1 / (1 + math.exp(-3 * d_e)) - self.damping) - self.mult_ang * ka * d_th),
                    sign * self.scale * (self.mult_lin * (1 / (1 + math.exp(-3 * d_e)) - self.damping) + self.mult_ang * ka * d_th)]

    def adjustAngleRange(self, rad):
      while rad > math.pi:
        rad -= 2*math.pi
      while rad < -math.pi:
        rad += 2*math.pi
      return rad
