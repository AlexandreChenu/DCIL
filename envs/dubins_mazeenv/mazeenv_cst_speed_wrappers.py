import sys
import os
sys.path.append(os.getcwd())
# from .mazeenv import *
from .mazeenv_cst_speed import *
from matplotlib import collections  as mc
from matplotlib.patches import Circle
import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch

from .skill_manager_mazeenv import *

import pdb


class DubinsMazeEnvGCPHERSB3(DubinsMazeEnv):

    def __init__(self, L_full_demonstration,
                        L_states,
                        L_inner_states,
                        L_goals,
                        L_budgets, mazesize = "5", do_overshoot = True):

        args={
                'mazesize':int(mazesize),
                'random_seed':0,
                'mazestandard':False,
                'wallthickness':0.1,
                'wallskill':True,
                'targetkills':True,
                'max_steps': 50,
                'width': 0.1
            }

        super(DubinsMazeEnvGCPHERSB3,self).__init__(args = args)

        ## init the skill manager
        self.skill_manager = SkillsManager(L_full_demonstration,
                                    L_states, ## starting states
                                    L_inner_states, ## starting states also (Mujoco format)
                                    L_goals, ## skill-goals
                                    L_budgets) ## skill length in time-steps
        self.args = args



        # Max steps per episode and counter
        self.max_steps = args['max_steps']
        self.rollout_steps = 0

        # Action space and observation space
        self.action_space = spaces.Box(np.array([-1.]), np.array([1.]), dtype = np.float32)
        ms = int(args['mazesize'])

        self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(
                        low = np.array([0,0,-4]),
                        high = np.array([ms,ms,4])),
                    "achieved_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                    "desired_goal": spaces.Box(
                        low = np.array([0,0]),
                        high = np.array([ms,ms])),
                }
            )

        self.state =  np.array([0.5, 0.5, 0.])
        self.goal= np.array([args['mazesize']-0.5, args['mazesize']-0.5])

        self.traj = []

        self.testing = False
        self.expanded = False
        self.target_reached = False
        self.overshoot = False
        self.do_overshoot = do_overshoot

        self.buffer_transitions = []

        self.frame_skip = 2
        self.max_reward = 1.
        self.width_success = 0.1
        self.total_steps = 0


    def state_act(self, action):
        r = np.copy(self.state)
        return self.update_state(r, action, self.delta_t)

    def compute_distance_in_goal_space(self, goal1, goal2):
        """
        distance in the goal space
        """
        goal1 = np.array(goal1)
        goal2 = np.array(goal2)

        if len(goal1.shape) ==  1:
            return np.linalg.norm(goal1 - goal2, axis=-1)
        else:
            return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        compute the reward according to distance in goal space
        R \in {0,1}
        """
        ### single goal
        if len(achieved_goal.shape) ==  1:
            dst = self.compute_distance_in_goal_space(achieved_goal, desired_goal)

            _info = {'reward_boolean': dst<= self.width_success}

            if _info['reward_boolean']:
                return self.max_reward
            else:
                return 0.

        ### tensor of goals
        else:

            distances = self.compute_distance_in_goal_space(achieved_goal, desired_goal)
            distances_mask = (distances <= self.width_success).astype(np.float32)

            rewards = distances_mask*self.max_reward

            return rewards


    def step(self, action) :
        """
        step of the environment

        3 cases:
            - target reached
            - time limit
            - else
        """

        state = self.get_state()

        ## enable frame skipping
        for i in range(self.frame_skip):
            new_state, reward, done, info = self._step(action)
            new_inner_state = new_state.copy()

            self.traj.append(new_state)

        ## walls kill or not?
        # done = not info["valid_action"]
        done = False

        self.rollout_steps += 1

        dst = np.linalg.norm(self.project_to_goal_space(new_state) - self.goal)
        info = {'target_reached': dst<= self.width_success}

        info['goal_indx'] = copy.deepcopy(self.skill_manager.indx_goal)
        info['goal'] = copy.deepcopy(self.goal)

        ### Case 1 - target reached
        if info['target_reached']: # achieved goal

            self.target_reached = True

            self.skill_manager.add_success(self.skill_manager.indx_goal)

            if self.skill_manager.skipping: ## stop skipping if overshooting
                self.skill_manager.skipping = False

            done = True
            reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

            prev_goal = self.goal.copy()
            info['done'] = done
            info['goal'] = self.goal.copy()
            info['traj'] = self.traj

            if self.overshoot:
                info['overshoot_success'] = True

            return OrderedDict([
                    ("observation", new_state.copy()), ## TODO: what's the actual state?
                    ("achieved_goal", self.project_to_goal_space(new_state).copy()),
                    ("desired_goal", prev_goal)]), reward, done, info

        ## Case 2 - Time limit
        elif self.rollout_steps >= self.max_steps:

            self.target_reached = False

            ## add failure to skill results
            self.skill_manager.add_failure(self.skill_manager.indx_goal)

            reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

            done = True
            prev_goal = self.goal.copy()
            info['done'] = done
            info['goal'] = self.goal.copy()
            info['traj'] = self.traj

            ## time limit for SB3s
            info["TimeLimit.truncated"] = True

            return OrderedDict([
                    ("observation", new_state.copy()),
                    ("achieved_goal", self.project_to_goal_space(new_state).copy()),
                    ("desired_goal", prev_goal)]), reward, done, info

        else:
            if done:
                info['traj'] = self.traj
                info['goal'] = self.goal.copy()

            self.target_reached = False

            reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

            info['done'] = done
            return OrderedDict([
                    ("observation", new_state.copy()),
                    ("achieved_goal", self.project_to_goal_space(new_state).copy()),
                    ("desired_goal", self.goal.copy()),]), reward, done, info

    def step_test(self, action) :
        """
        step method for evaluation -> no reward computed, no time limit etc.
        """
        for i in range(self.frame_skip):
            new_state, reward, done, info = self._step(action)
            self.traj.append(new_state)

        self.rollout_steps += 1

        dst = np.linalg.norm(self.project_to_goal_space(new_state) - self.goal)
        info = {'target_reached': dst<= self.width_success}

        reward = 0.

        return OrderedDict([
        ("observation", new_state.copy()),
        ("achieved_goal", self.project_to_goal_space(new_state).copy()),
        ("desired_goal", self.goal.copy()),]), reward, done, info

    def get_state(self):
        return self.state

    def _get_obs(self):

        state = self.get_state()
        achieved_goal = self.project_to_goal_space(state)

        return OrderedDict(
        [
        ("observation", state.copy()),
        ("achieved_goal", achieved_goal.copy()),
        ("desired_goal", self.goal.copy()),
        ]
        )

    def goal_vector(self):
        return self.goal

    def set_state(self, state):
        self.state = np.array(state)
        return self.get_state()

    def set_goal_state(self, goal_state):
        self.goal_state = np.array(goal_state)
        self.goal = self.project_to_goal_space(goal_state)
        return 0

    def project_to_goal_space(self, state):
        """
        Env-dependent projection of a state in the goal space.
        In a mazeenv -> keep (x,y) coordinates
        """
        return np.array(state[:2])

    def select_skill(self):
        """
        Sample skill using skill manager.
        """
        return self.skill_manager.select_skill()

    def reset_skill_by_nb(self, skill_nb):
        """
        Reset agent to the starting state of a given skill
        """
        self.reset()

        starting_state, length_skill, goal_state = self.skill_manager.get_skill(skill_nb)

        self.set_goal_state(goal_state)
        self.set_state(starting_state)
        self.max_steps = length_skill
        return

    def next_skill(self):
        """
        Shift to the next skill.
        Update goal, rollout step counter and max steps.
        """
        # print("self.goal = ", self.goal)
        goal_state, length_skill, skill_avail = self.skill_manager.next_skill()

        if skill_avail:
            self.set_goal_state(goal_state)
            self.max_steps = length_skill
            self.rollout_steps = 0
        return skill_avail


    def reset(self, eval = False):
        """
        Reset environment.

        2 cases:
            - reset after success -> try to overshoot
                    if a following skill exists -> overshoot i.e. update goal, step counter
                    and budget but not the current state
                    else -> reset to a new skill
            - reset to new skill
        """
        ## Case 1 - success -> automatically try to overshoot
        if self.target_reached and self.do_overshoot:
            prev_goal = self.goal.copy()
            self.subgoal = self.goal.copy()
            skill_avail = self.next_skill()

            ## shift to a next skill is possible (last skill not reached)
            if skill_avail:
                state = self.get_state()
                self.overshoot = True

                return OrderedDict([
                        ("observation", state.copy()),
                        ("achieved_goal", self.project_to_goal_space(state).copy()),
                        ("desired_goal", self.goal.copy()),])

            ## shift impossible (current skill is last one)
            else:
                self.overshoot = False
                self.target_reached = False
                out_state = self.reset()
                return out_state

        ## Case 2 - no success: reset to new skill
        else:
            self.testing = False
            self.skipping = False
            self.skill_manager.skipping = False
            self.overshoot = False

            ## sample a skill
            starting_state, length_skill, goal_state = self.select_skill()

            self.set_goal_state(goal_state)
            self.set_state(starting_state)

            self.max_steps = length_skill
            #self.max_steps = 10

            self.rollout_steps = 0
            self.traj = []
            state = self.get_state()
            self.traj.append(state)

            return OrderedDict([
                    ("observation", state.copy()),
                    ("achieved_goal", self.project_to_goal_space(state).copy()),
                    ("desired_goal", self.goal.copy()),])
