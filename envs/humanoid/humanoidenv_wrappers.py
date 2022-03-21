import sys
import os
sys.path.append(os.getcwd())
#from gym_marblemaze.envs.mazeenv.maze.maze import Maze
# from gym_marblemaze.envs.dubins_mazeenv.mazeenv import *
# from gym_marblemaze.envs.mazeenv.mazeenv import *
#from maze.maze import Maze


from .humanoidenv import MyHumanoidEnv
from .skill_manager_humanoid import SkillsManager

from matplotlib import collections  as mc
from matplotlib.patches import Circle
import gym
from gym import error, spaces
from gym.utils import seeding

gym._gym_disable_underscore_compat = True

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch

import  pdb


class HumanoidEnvGCPHERSB3(gym.Env#MyComplexFetchEnv):
	):

	def __init__(self, L_full_demonstration, L_full_inner_demonstration, L_states, starting_states, starting_inner_states, L_actions, L_full_observations, L_goals, L_inner_states, L_budgets, m_goals = None, std_goals = None, env_option = "", do_overshoot=True):

		self.env = MyHumanoidEnv()

		## skills
		self.skill_manager = SkillsManager(L_full_demonstration, L_full_inner_demonstration, L_states, L_actions, L_full_observations, L_goals, L_inner_states, L_budgets, m_goals, std_goals , env_option)

		self.max_steps = 50
		# Counter of steps per episode
		self.rollout_steps = 0

		self.action_space = self.env.env.action_space

		self.env_option = env_option

		self.incl_extra_full_state = 1

		self.m_goals = m_goals
		self.std_goals = std_goals

		###TODO: change observation space
		if "com" in self.env_option :
			self.observation_space = spaces.Dict(
					{
						"observation": gym.spaces.Box(-3., 3., shape=(378,), dtype='float32'),
						"achieved_goal": spaces.Box(
							low = np.array([-3,-3,-3]),
							high = np.array([3,3,3])), # gripper_pos + object pos
						"desired_goal": spaces.Box(
							low = np.array([-3,-3,-3]),
							high = np.array([3,3,3])),
					}
				)

		else:
			self.observation_space = spaces.Dict(
					{
						"observation": gym.spaces.Box(-3., 3., shape=(378,), dtype='float32'),
						"achieved_goal": spaces.Box(
							low = np.array([-3,-3,-3]),
							high = np.array([3,3,3])),
						"desired_goal": spaces.Box(
							low = np.array([-3,-3,-3]),
							high = np.array([3,3,3])),
					}
				)


		# self.width_success = width_success ## hyper-parametre assez difficile à ajuster en dim 6
		self.width_success = 0.075


		self.total_steps = sum(self.skill_manager.L_budgets)


		self.traj = []

		# self.indx_start = 0
		# self.indx_goal = -1
		self.testing = False
		self.expanded = False

		self.buffer_transitions = []

		self.bonus = True
		self.weighted_selection = True

		self.target_selection = False
		self.target_ratio = 0.3

		self.frame_skip = 1
		# self.frame_skip = 3

		self.target_reached = False
		self.overshoot = False

		self.max_reward = 1.

	def compute_distance_in_goal_space(self, in_goal1, in_goal2):
		"""
		goal1 = achieved_goal
		goal2 = desired_goal
		"""

		goal1 = copy.deepcopy(in_goal1)
		goal2 = copy.deepcopy(in_goal2)

		if "com" in self.env_option:

			if len(goal1.shape) ==  1:
				euclidian_goal1 = goal1[:3]
				euclidian_goal2 = goal2[:3]

				return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)

			else:
				euclidian_goal1 = goal1[:,:3]
				euclidian_goal2 = goal2[:,:3]

				return np.linalg.norm(euclidian_goal1[:,:] - euclidian_goal2[:,:], axis=-1)

		else:
			if len(goal1.shape) ==  1:
				return np.linalg.norm(goal1 - goal2, axis=-1)
			else:
				return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)


	def compute_reward(self, achieved_goal, desired_goal, info):
		"""
        compute the reward according to distance in goal space
        R \in {0,10}
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
			## compute -1 or 0 reward
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
		state = self.env.get_state()

		for i in range(self.frame_skip):
			new_state, reward, done, info = self.env.step(action)

			new_inner_state = self.env.get_restore()

			self.traj.append(new_state[:3])

		self.rollout_steps += 1

		dst = self.compute_distance_in_goal_space(self.project_to_goal_space(new_state),  self.goal)
		info = {'target_reached': dst<= self.width_success}

		info['goal_indx'] = copy.deepcopy(self.skill_manager.indx_goal)
		info['goal'] = copy.deepcopy(self.goal)


		if info['target_reached']: # achieved goal

			self.target_reached = True

			self.skill_manager.add_success(self.skill_manager.indx_goal)

			done = True
			reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

			prev_goal = self.goal.copy()
			info['done'] = done
			info['goal'] = self.goal.copy()
			info['traj'] = self.traj

			if self.overshoot:
				info['overshoot_success'] = True
				self.skill_manager.add_success_overshoot(self.skill_manager.indx_goal)
			# self.reset()

			return OrderedDict([
					("observation", new_state.copy()), ## TODO: what's the actual state?
					("achieved_goal", self.project_to_goal_space(new_state).copy()),
					("desired_goal", prev_goal)]), reward, done, info

		elif self.rollout_steps >= self.max_steps:
			### failed skill
			self.target_reached = False

			## add failure to skill results
			self.skill_manager.add_failure(self.skill_manager.indx_goal)

			reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

			done = True ## no done signal if timeout (otherwise non-markovian process)

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

			done = False
			info['done'] = done
			self.target_reached = False

			reward = self.compute_reward(self.project_to_goal_space(new_state), self.goal, info)

			return OrderedDict([
					("observation", new_state.copy()),
					("achieved_goal", self.project_to_goal_space(new_state).copy()),
					("desired_goal", self.goal.copy()),]), reward, done, info


	def step_test(self, action) :
		"""
        step method for evaluation -> no reward computed, no time limit etc.
        """

		for i in range(self.frame_skip):
			new_state, reward, done, info = self.env.step(action)

			self.traj.append(new_state[:3])

		self.rollout_steps += 1

		dst = np.linalg.norm(self.project_to_goal_space(new_state) - self.goal)
		info = {'target_reached': dst<= self.width_success}
		#reward = -dst

		reward = 0.

		return OrderedDict([
				("observation", new_state.copy()),
				("achieved_goal", self.project_to_goal_space(new_state).copy()),
				("desired_goal", self.goal.copy()),]), reward, done, info


	def _get_obs(self):

		state = self.env.get_state()
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

	def set_state(self, inner_state):
		# print("inner_state = ", inner_state)
		# self.env.restore(inner_state)
		self.env.env.set_inner_state(inner_state)

	def set_goal_state(self, goal_state):
		self.goal_state = goal_state
		self.goal = self.project_to_goal_space(goal_state)
		return 0


	def project_to_goal_space(self, state):
		"""
        Env-dependent projection of a state in the goal space.
        In a humanoidenv -> keep (x,y,z) coordinates of the torso
        """

		com_pos = self.get_com_pos(state)
		com_vel = self.get_com_vel(state)

		#norm_gripper_velp = np.linalg.norm(gripper_velp)

		if "vel" in self.env_option:
			return np.concatenate((np.array(com_pos), np.array(com_vel)))

		else:
			return np.array(com_pos)

	## TODO: get coordinates in state
	def get_com_pos(self, state):
		"""
		get center of mass position from full state for torso?
		"""
		# print("len(list(state)) = ", len(list(state)))
		assert len(list(state))== 378

		com_pos = state[:3]
		# gripper_pos = state[102:105]

		assert len(list(com_pos)) == 3

		return com_pos

	def get_com_vel(self, state):
		"""
		get center of mass velocity from full state for torso
		"""
		assert len(list(state)) == 378

		object_pos = state[105:108]
		# object_pos = state[123:126]
		assert len(list(object_pos))==3

		# print("indx object pos = ", indx)

		return object_pos

	def select_skill(self):
		"""
		Sample skill for low-level policy training.

		"""
		return self.skill_manager.select_skill()

	def reset_skill_by_nb(self, skill_nb):

		self.env.reset()

		starting_state, length_skill, goal_state = self.skill_manager.get_skill(skill_nb)

		self.set_goal_state(goal_state)
		self.set_state(starting_state)
		self.max_steps = length_skill
		return

	def next_skill(self):
		goal_state, length_skill, advance_bool = self.skill_manager.next_skill()

		if advance_bool:

			self.set_goal_state(goal_state)
			self.max_steps = length_skill
			self.rollout_steps  = 0

		return advance_bool


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
		if self.target_reached: ## automatically overshoot
			self.subgoal = self.goal.copy()
			advance_bool = self.next_skill()
			self.target_reached = False

            ## shift to a next skill is possible (last skill not reached)
			if advance_bool:
				# pdb.set_trace()
				state = copy.deepcopy(self.env.get_state())
				self.overshoot = True

				return OrderedDict([
						("observation", state.copy()),
						("achieved_goal", self.project_to_goal_space(state).copy()),
						("desired_goal", self.goal.copy()),])


			## shift impossible (current skill is the last one)
			else:
				#pdb.set_trace()
				self.overshoot = False
				self.target_reached = False
				out_state = self.reset()
				return out_state

		## Case 2 - no success: reset to new skill
		else:
			# print("true reset")
			self.env.reset()

			self.testing = False
			self.skipping = False
			self.skill_manager.skipping = False

			self.overshoot = False

			starting_state, length_skill, goal_state = self.select_skill()

			self.set_goal_state(goal_state)
			self.set_state(starting_state)

			self.max_steps = length_skill

			self.rollout_steps = 0
			self.traj = []

			state = self.env.get_state()

			self.traj.append(state[:3])

			return OrderedDict([
					("observation", state.copy()),
					("achieved_goal", self.project_to_goal_space(state).copy()),
					("desired_goal", self.goal.copy()),])
