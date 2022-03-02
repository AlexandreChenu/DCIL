import sys
import os
sys.path.append(os.getcwd())
from .maze.maze import Maze
#from maze.maze import Maze
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib.patches import Circle
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import math
import random
import copy

class State(object):
    def __init__(self, lst=None):
        if lst is not None:
            self.x = lst[0]
            self.y = lst[1]
        else:
            self.x = 0.5
            self.y = 0.5

    def to_list(self):
        return np.array([self.x, self.y])

    def distance_to(self,other):
        dx = self.x-other.x
        dy = self.y-other.y
        return math.sqrt(dx*dx+dy*dy)

    def act(self,action):
        a = action[:]
        if a[0]>.1:
            a[0]=.1
        if a[0]<-.1:
            a[0]=-.1
        if a[1]>.1:
            a[1]=.1
        if a[1]<-.1:
            a[1]=-.1
        r = State()
        r.x = self.x + a[0]
        r.y = self.y + a[1]
        return r

    def perturbation(self,mag=1e-5):
        r = State()
        r.x = self.x + mag * random.uniform(0,1)
        r.y = self.y + mag * random.uniform(0,1)
        return r

    def isInBounds(self,maze):
        return (self.x>0 and self.y>0 and self.x<maze.num_cols and self.y<maze.num_rows)

    def __str__(self):
        return "({:10.2f},{:10.2f})".format(self.x,self.y)

MAX_STEER = np.pi
MAX_ANGLE = 2*np.pi
MIN_ANGLE = -2*np.pi
MAX_SPEED = 0.5
MAX_ACCELERATION = 1.
MIN_ACCELERATION = 0.
THRESHOLD_DISTANCE_2_GOAL = 0.02
MAX_X = 10.
MAX_Y = 10.
max_ep_length = 800

# Vehicle parameters
LENGTH = 0.45  # [m]
WIDTH = 0.2  # [m]
BACKTOWHEEL = 0.1  # [m]
WHEEL_LEN = 0.03  # [m]
WHEEL_WIDTH = 0.02  # [m]
TREAD = 0.07  # [m]
WB = 0.45  # [m]

class DubinsMazeEnv(Maze, gym.Env):
    def __init__(self,args={
            'mazesize':5,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True
        }):
        self.setup(args)
        self.allsteps = []
        self.lines = None
        self.interacts = 0

    def setup(self,args):
        super(DubinsMazeEnv,self).__init__(args['mazesize'],args['mazesize'],seed=args['random_seed'],standard=args['mazestandard'])
        ms = int(args['mazesize'])
        self.state =  np.array([0.5, 0.5, 0.])
        self.steps = []
        self.obs_dim = 2
        self.thick = args['wallthickness']

        self.action_space = spaces.Box(np.array([-1.]), np.array([1.]), dtype = np.float32)
        low = np.array([0.,0.,-4.])
        high = np.array([ms,ms,4.])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.metadata = dict()
        self.reward_range = [0.0, 2.0]
        #self.unwrapped = True
        self._configured = False

        self.alive_bonus=0.001
        self.distance_bonus=0.01
        self.obstacle_reward=0
        # self.obstacle_reward = 0
        self.target_reward=1
        #self.wallskill=args['wallskill']
        #self.targetkills=args['targetkills']
        self.wallskill = False
        self.targetkills = False


        self.goal = np.array([2.5, 0.5])
        self.width = 0.1

        self.delta_t = 0.2

        self.frame_skip = 1


    def reward_function(self, obs, goal):
        d =  np.linalg.norm(obs[:2] - goal[:2], axis=-1)

        if d > 0.10:
            return 1./d
        else:
            return 10.


    def update_state(self, state, a, DT):
        # print("Updating state")

        steer = a[0]

        if steer >= MAX_STEER:
            steer = MAX_STEER
        elif steer <= -MAX_STEER:
            steer = -MAX_STEER

        state[0] = state[0] + MAX_SPEED*math.cos(state[2]) * DT
        state[1] = state[1] + MAX_SPEED*math.sin(state[2]) * DT
        new_orientation = state[2] + steer * DT

        ## check limit angles
        if ((new_orientation <= MAX_ANGLE) and (new_orientation >= MIN_ANGLE)):
            state[2] = new_orientation

        return state

    def state_act(self, action):
        r = np.copy(self.state)
        return self.update_state(r, action, self.delta_t)

    def state_perturbation(self, mag=1e-5):
        r = np.copy(self.state)
        r[0] = r[0] + mag * random.uniform(0,1)
        r[1] = r[1] + mag * random.uniform(0,1)
        return r

    def state_isInBounds(self, state, maze):

        return (state[0]>0 and state[1]>0 and state[0] < maze.num_cols and state[1] < maze.num_rows)

    def seed(self,v):
        pass

    def reset(self):
        return self.reset_primitive()

    def reset_primitive(self):
        self.state = np.array([0.5, 0.5, 0.])

        return self.state

    def set_state(self,state):
        self.state =  np.array([0.5, 0.5, 0.])

    def close(self):
        pass

    @property
    def dt(self):
        return 0.01

    def state_vector(self):
        return self.state

    def __seg_to_bb(self,s):
        bb = [list(s[0]),list(s[1])]
        if bb[0][0] > bb[1][0]:
            bb[0][0] = s[1][0]
            bb[1][0] = s[0][0]
        if bb[0][1] > bb[1][1]:
            bb[0][1] = s[1][1]
            bb[1][1] = s[0][1]
        return bb

    def __bb_intersect(self,a,b,e=1e-8):
        return (a[0][0] <= b[1][0] + e
                and a[1][0] + e >= b[0][0]
                and a[0][1] <= b[1][1] + e
                and a[1][1] + e >= b[0][1]);

    def __cross(self,a,b):
        return a[0]*b[1] - a[1]*b[0]

    def __is_point_right_of_line(self,a,b):
        atmp=[a[1][0] - a[0][0], a[1][1] - a[0][1]];
        btmp=[b[0] - a[0][0], b[1] - a[0][1]];
        return self.__cross(atmp,btmp) < 0;

    def __is_point_on_line(self,a,b,e=1e-8):
        atmp=[a[1][0] - a[0][0], a[1][1] - a[0][1]];
        btmp=[b[0] - a[0][0], b[1] - a[0][1]];
        return self.__cross(atmp,btmp) <= 1e-8;

    def __segment_touches_or_crosses_line(self,a,b,e=1e-8):
        return (self.__is_point_on_line(a,b[0],e)
            or self.__is_point_on_line(a,b[1],e)
            or (self.__is_point_right_of_line(a,b[0])
                != self.__is_point_right_of_line(a,b[1])))

    def __segments_intersect(self,a,b,e=1e-8):
        return (self.__bb_intersect(self.__seg_to_bb(a),self.__seg_to_bb(b),e)
            and self.__segment_touches_or_crosses_line(a,b,e)
            and self.__segment_touches_or_crosses_line(b,a,e))

    def state_in_wall(self,s):
        t = self.thick
        if t==0:
            return False
        def in_hwall(i,j,t=0):
            return s[0]>=i-t and s[0]<=i+t and s[1]>=j and s[1]<=j+1
        def in_vwall(i,j,t=0):
            return s[0]>=i and s[0]<i+1 and s[1]>=j-t and s[1]<=j+t
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j].walls["top"]:
                    if in_hwall(i,j,t):
                        return True
                if self.grid[i][j].walls["bottom"]:
                    if in_hwall(i+1,j,t):
                        return True
                if self.grid[i][j].walls["left"]:
                    if in_vwall(i,j,t):
                        return True
                if self.grid[i][j].walls["right"]:
                    if in_vwall(i,j+1,t):
                        return True
        return False

    def random_state(self):
        while True:
            s = self.num_cols * np.random.rand(2)
            if not self.state_in_wall(s):
                break
        return s

    def valid_action(self,action,cur_state=None):
        if cur_state is None:
            cur_state = np.copy(self.state)
        if len(action)==len(list(self.action_space.sample())):
            #if not cur_state.act(action).isInBounds(self):
            if not self.state_isInBounds(self.state_act(action), self):
                return False
            sa = self.state_act(action)
            #print("cur_state = ", cur_state)
            #print("sa = ", sa)

            #s = [(cur_state.x,cur_state.y),(sa.x,sa.y) ]
            s = [(cur_state[0],cur_state[1]),(sa[0],sa[1]) ]
            if self.lines is None:
                self.lines = [] #todo optim
                def add_hwall(lines,i,j,t=0):
                    lines.append([(i-t,j),(i-t,j+1)])
                    if t>0:
                        lines.append([(i-t,j+1),(i+t,j+1)])
                        lines.append([(i+t,j),(i+t,j+1)])
                        lines.append([(i+t,j),(i-t,j)])
                def add_vwall(lines,i,j,t=0):
                    lines.append([(i,j-t),(i+1,j-t)])
                    if t>0:
                        lines.append([(i+1,j-t),(i+1,j+t)])
                        lines.append([(i,j+t),(i+1,j+t)])
                        lines.append([(i,j+t),(i,j-t)])
                t = self.thick
                for i in range(len(self.grid)):
                    for j in range(len(self.grid[i])):
                        if self.grid[i][j].walls["top"]:
                            add_hwall(self.lines,i,j,t)
                        if self.grid[i][j].walls["bottom"]:
                            add_hwall(self.lines,i+1,j,t)
                        if self.grid[i][j].walls["left"]:
                            add_vwall(self.lines,i,j,t)
                        if self.grid[i][j].walls["right"]:
                            add_vwall(self.lines,i,j+1,t)
            for l in self.lines:
                if self.__segments_intersect(l,s):
                    return False
            return True
        else:
            return False

    def step(self,action):
        for i in range(self.frame_skip):
            new_state, reward, done, info = self._step(action)
        if info["valid_action"]==False:
            done = True
        return new_state, reward, done, info

    def setConfig(self,args):
        self.alive_bonus = float(args['alivebonus'])
        self.distance_bonus = float(args['distancebonus'])
        self.obstacle_reward=float(args['obstaclereward'])
        self.target_reward=float(args['targetreward'])


    def _step(self,action):

        if np.array(action).shape==(1,3):
            action = action[0]

        # print("action = ", action)

        if not self.valid_action(action):
            #DEBUG : invalid actions are simply ignored
            #raise NotImplementedError #todo
            #self.state = self.state_perturbation(1e-9)
            reward = self.reward_function(self.state, self.goal)
            #return self.state.to_list(), self.obstacle_reward, self.wallskill, {'target_reached': False}
            return list(self.state), reward + self.obstacle_reward, self.wallskill, {'target_reached': False, 'valid_action': False}

        if self.valid_action(action):
            state_before=self.state
            #self.state = self.state.act(action)
            self.state = self.state_act(action)
            self.interacts += 1
            self.steps.append([state_before, action, self.state])

        #allsteps is obsolete ?
        self.allsteps.append([state_before, action, self.state])
        if len(self.allsteps)>10000:
            self.allsteps = self.allsteps[1:]

        reward = self.reward_function(self.state, self.goal)

        #done = True if dst<.2 and self.targetkills else False
        done = False # env never terminates

        #info = {'target_reached': dst<.2}
        info = {'valid_action': True}

        #return self.state.to_list(),reward,done,info
        return list(self.state),reward,done,info


    def draw(self,ax,color=None,**kwargs):
        Maze.draw(self,ax,thick=self.thick)
        if self.num_cols==2:
            target = [1-.5,self.num_cols-.5]
        else:
            target = [self.num_rows-.5,self.num_cols-.5]
        #c = Circle(target,.2,fill=False)
        #ax.add_patch(c)
        if 'paths' in kwargs and kwargs['paths']:
            print("Drawing ",len(self.allsteps)," steps")
            lines = []
            colors = []
            s = len(self.allsteps)
            i = 0
            for b,ac,a in self.allsteps:
                lines.append([(b.x,b.y),(a.x,a.y)])
                if color is None:
                    colors.append([float(i)/s,0,0])
                else:
                    colors.append(color)
                i += 1
            lc = mc.LineCollection(lines, linewidths=1, colors=colors)
            ax.add_collection(lc)
            print("  ... done")

    def plot_car(self, state, ax, alpha , cabcolor="-r", truckcolor="-k"):  # pragma: no cover
        # print("Plotting Car")
        # x = self.pose[0]*MAX_X #self.pose[0]
        # y = self.pose[1]*MAX_Y #self.pose[1]
        # yaw = self.pose[2] #self.pose[2]
        # steer = self.action[1]*MAX_STEER #self.action[1]

        x = state[0] #self.pose[0]
        y = state[1] #self.pose[1]
        yaw = state[2] #self.pose[2]

        length = 0.2  # [m]
        width = 0.1  # [m]
        backtowheel = 0.05  # [m]
        # WHEEL_LEN = 0.03  # [m]
        # WHEEL_WIDTH = 0.02  # [m]
        # TREAD = 0.07  # [m]
        wb = 0.45  # [m]

        outline = np.array([[-backtowheel, (length - backtowheel), (length - backtowheel), -backtowheel, -backtowheel],
        					[width / 2, width / 2, - width / 2, -width / 2, width / 2]])

        # fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
        # 					 [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])
        #
        # rr_wheel = np.copy(fr_wheel)

        # fl_wheel = np.copy(fr_wheel)
        # fl_wheel[1, :] *= -1
        # rl_wheel = np.copy(rr_wheel)
        # rl_wheel[1, :] *= -1

        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
        				 [-math.sin(yaw), math.cos(yaw)]])
        # Rot2 = np.array([[math.cos(steer), math.sin(steer)],
        # 				 [-math.sin(steer), math.cos(steer)]])

        # fr_wheel = (fr_wheel.T.dot(Rot2)).T
        # fl_wheel = (fl_wheel.T.dot(Rot2)).T
        # fr_wheel[0, :] += WB
        # fl_wheel[0, :] += WB

        # fr_wheel = (fr_wheel.T.dot(Rot1)).T
        # fl_wheel = (fl_wheel.T.dot(Rot1)).T

        outline = (outline.T.dot(Rot1)).T
        # rr_wheel = (rr_wheel.T.dot(Rot1)).T
        # rl_wheel = (rl_wheel.T.dot(Rot1)).T

        outline[0, :] += x
        outline[1, :] += y
        # fr_wheel[0, :] += x
        # fr_wheel[1, :] += y
        # rr_wheel[0, :] += x
        # rr_wheel[1, :] += y
        # fl_wheel[0, :] += x
        # fl_wheel[1, :] += y
        # rl_wheel[0, :] += x
        # rl_wheel[1, :] += y

        ax.plot(np.array(outline[0, :]).flatten(),
        		 np.array(outline[1, :]).flatten(), color=truckcolor, alpha = alpha)
        # plt.plot(np.array(fr_wheel[0, :]).flatten(),
        # 		 np.array(fr_wheel[1, :]).flatten(), truckcolor)
        # plt.plot(np.array(rr_wheel[0, :]).flatten(),
        # 		 np.array(rr_wheel[1, :]).flatten(), truckcolor)
        # plt.plot(np.array(fl_wheel[0, :]).flatten(),
        # 		 np.array(fl_wheel[1, :]).flatten(), truckcolor)
        # plt.plot(np.array(rl_wheel[0, :]).flatten(),
        # 		 np.array(rl_wheel[1, :]).flatten(), truckcolor)
        #plt.scatter(x, y, marker = ".")


class FreeDubinsMazeEnv(DubinsMazeEnv, gym.Env):
    def __init__(self,args={
            'mazesize':15,
            'random_seed':0,
            'mazestandard':False,
            'wallthickness':0.1,
            'wallskill':True,
            'targetkills':True
        }):
        super(FreeDubinsMazeEnv,self).__init__(args = args)
        self.empty_grid()
        #self.no_grid()
