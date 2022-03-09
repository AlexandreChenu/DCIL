import sys
import os
sys.path.append(os.getcwd())

from .fetch_env import MyComplexFetchEnv
from .skill_manager_fetchenv import SkillManager

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


class ComplexFetchEnvGCPHERSB3(gym.Env):


	def __init__(self, L_full_demonstration, L_full_inner_demonstration, L_states, starting_states, starting_inner_states, L_actions, L_full_observations, L_goals, L_inner_states, L_budgets, m_goals = None, std_goals = None, env_option = "", do_overshoot=True):

		self.env = MyComplexFetchEnv()

		## skill manager
		self.skill_manager = SkillManager(L_full_demonstration, L_full_inner_demonstration, L_states, starting_states, starting_inner_states, L_actions, L_full_observations, L_goals, L_inner_states, L_budgets, m_goals, std_goals , env_option)

		self.max_steps = 50
		# Counter of steps per episode
		self.rollout_steps = 0

		self.action_space = self.env.env.action_space

		self.env_option = env_option

		self.incl_extra_full_state = 0

		self.m_goals = m_goals
		self.std_goals = std_goals

		if "full" in self.env_option :
			self.observation_space = spaces.Dict(
					{
						"observation": gym.spaces.Box(-3., 3., shape=(268 + 336 * self.incl_extra_full_state,), dtype='float32'),
						"achieved_goal": spaces.Box(
							low = np.array([-3,-3,-3,-3,-3,-3]),
							high = np.array([3,3,3,3,3,3])), # gripper_pos + object pos
						"desired_goal": spaces.Box(
							low = np.array([-3,-3,-3,-3,-3,-3]),
							high = np.array([3,3,3,3,3,3])),
					}
				)

		elif "grasping" in self.env_option:
			self.observation_space = spaces.Dict(
					{
						"observation": gym.spaces.Box(-3., 3., shape=(268 + 336 * self.incl_extra_full_state,), dtype='float32'),
						"achieved_goal": spaces.Box(
							low = np.array([-3,-3,-3,0.]),
							high = np.array([3,3,3,1.])), # gripper_pos + object pos
						"desired_goal": spaces.Box(
							low = np.array([-3,-3,-3,0.]),
							high = np.array([3,3,3,1.])),
					}
				)

		else:
			self.observation_space = spaces.Dict(
					{
						"observation": gym.spaces.Box(-3., 3., shape=(268 + 336 * self.incl_extra_full_state,), dtype='float32'),

						"achieved_goal": spaces.Box(
							low = np.array([-3,-3,-3]),
							high = np.array([3,3,3])), # gripper_pos
						"desired_goal": spaces.Box(
							low = np.array([-3,-3,-3]),
							high = np.array([3,3,3])),
					}
				)


		# self.width_reward = width_reward ## hyper-parametre assez difficile à ajuster en dim 6
		self.width_success = 0.05


		self.total_steps = sum(self.skill_manager.L_budgets)

		self.traj_gripper = []
		self.traj_object = []

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
		self.do_overshoot = do_overshoot

		self.max_reward = 1.


	def compute_distance_in_goal_space(self, in_goal1, in_goal2):
		"""
		goal1 = achieved_goal
		goal2 = desired_goal
		"""

		goal1 = copy.deepcopy(in_goal1)
		goal2 = copy.deepcopy(in_goal2)

		if "grasping" in self.env_option:

			## single goal
			if len(goal1.shape) ==  1:
				euclidian_goal1 = goal1[:3]
				euclidian_goal2 = goal2[:3]

				if goal1[3] == goal2[3]: ## grasping boolean
					return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)

				else:
					return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1) + 1000000 ## if no grasping, artificially push goals far away
					# return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)
			## tensor of goals
			else:
				euclidian_goal1 = goal1[:,:3]
				euclidian_goal2 = goal2[:,:3]

				goal1_bool = goal1[:,3]
				goal2_bool = goal2[:,3]

				grasping_penalty = ((goal1_bool == goal2_bool).astype(np.float32)-1)*(-1000000)

				assert np.linalg.norm(euclidian_goal1[:,:] - euclidian_goal2[:,:], axis=-1).size == grasping_penalty.size

				return np.linalg.norm(euclidian_goal1[:,:] - euclidian_goal2[:,:], axis=-1) + grasping_penalty

		else:
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
				return self.max_reward#1.#0.
			else:
				return 0.#-1.

		### tensor of goals
		else:

			distances = self.compute_distance_in_goal_space(achieved_goal, desired_goal)
			distances_mask = (distances <= self.width_success).astype(np.float32)

			# rewards = distances_mask - 1. #- distances_mask * 0.1 # {-1, -0.1}
			rewards = distances_mask *self.max_reward

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

			new_inner_state = self.env.get_restore()[0]

			gripper_pos = self.get_gripper_pos(new_state)
			object_pos = self.get_object_pos(new_state) # best way to access object position so far

			self.traj_gripper.append(gripper_pos)
			self.traj_object.append(object_pos)

		#print("new_state.shape = ", new_state.shape)

		self.rollout_steps += 1

		dst = self.compute_distance_in_goal_space(self.project_to_goal_space(new_state),  self.goal)
		info = {'target_reached': dst<= self.width_success}

		info['goal_indx'] = copy.deepcopy(self.skill_manager.indx_goal)
		info['goal'] = copy.deepcopy(self.goal)

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
			info['traj'] = [self.traj_gripper, self.traj_object]

			if self.overshoot:
				info['overshoot_success'] = True
				self.skill_manager.add_success_overshoot(self.skill_manager.indx_goal)

			## add time to observation
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
			info['traj'] = [self.traj_gripper, self.traj_object]

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

			gripper_pos = self.get_gripper_pos(new_state)
			object_pos = self.get_object_pos(new_state) # best way to access object position so far

			self.traj_gripper.append(gripper_pos)
			self.traj_object.append(object_pos)

		self.rollout_steps += 1

		dst = np.linalg.norm(self.project_to_goal_space(new_state) - self.goal)
		info = {'target_reached': dst<= self.width_success}

		#reward = 0.

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
		self.env.env.set_inner_state(inner_state)

	def set_goal_state(self, goal_state):
		self.goal_state = goal_state
		self.goal = self.project_to_goal_space(goal_state)
		return 0

	def check_grasping(self, state):

		collision_l_gripper_link_obj = state[216 + 167]
		collision_r_gripper_link_obj = state[216 + 193]
		collision_object_table = state[216 + 67] ## add collision between object and table to improve grasing check

		## if quaternion angles
		# collision_l_gripper_link_obj = state[234 + 167]
		# collision_r_gripper_link_obj = state[234 + 193]
		# collision_object_table = state[234 + 67]

		if collision_l_gripper_link_obj and collision_r_gripper_link_obj and not collision_object_table :
			grasping = 1
		else:
			grasping = 0

		return grasping

	def project_to_goal_space(self, state):
		"""
		Env-dependent projection of a state in the goal space.
		In a fetchenv -> keep (x,y,z) coordinates of the gripper + 0,1 boolean
		if the object is grasped or not.
		"""

		#print("state.shape = ", state.shape)

		gripper_pos = self.get_gripper_pos(state)
		object_pos = self.get_object_pos(state)
		gripper_velp = self.get_gripper_velp(state)
		gripper_quat = self.get_gripper_quat(state)
		gripper_euler = self.get_gripper_euler(state)

		norm_gripper_velp = np.linalg.norm(gripper_velp)

		if "full" in self.env_option:
			return np.concatenate((np.array(gripper_pos), np.array(object_pos)))
		elif "grasping" in self.env_option:
			bool_grasping = self.check_grasping(state)
			return np.concatenate((np.array(gripper_pos), np.array([int(bool_grasping)])))
		else:
			return np.array(gripper_pos)

	def get_gripper_pos(self, state):
		"""
		get gripper position from full state
		"""
		# print("state = ", state)
		assert len(list(state))== 268 + 336 * self.incl_extra_full_state or len(list(state))== 268
		# assert len(list(state)) == 268 + 336 * self.incl_extra_full_state +1  or len(list(state))== 268

		gripper_pos = state[84:87]
		# gripper_pos = state[102:105]

		assert len(list(gripper_pos)) == 3

		return gripper_pos

	def get_object_pos(self, state):
		"""
		get object position from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		object_pos = state[105:108]
		# object_pos = state[123:126]
		assert len(list(object_pos))==3

		# print("indx object pos = ", indx)

		return object_pos

	def get_gripper_velp(self, state):
		"""
		get object position from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_velp = state[138:141]
		# gripper_velp = state[156:159]
		assert len(list(gripper_velp))==3

		return gripper_velp

	def get_gripper_quat(self, state):
		"""
		get object orientation from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_quat = state[36:40]
		assert len(list(gripper_quat))==4

		return gripper_quat

	def get_gripper_euler(self, state):
		"""
		get object orientation from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_quat = state[27:30] #TODO
		assert len(list(gripper_quat))==3

		return gripper_quat


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
		if self.target_reached and self.do_overshoot: ## automatically overshoot
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
				self.overshoot = False
				self.target_reached = False
				out_state = self.reset()
				return out_state

		## Case 2 - no success: reset to new skill
		else:
			self.env.reset()

			self.testing = False
			self.skipping = False
			self.skill_manager.skipping = False

			self.overshoot = False

			starting_state, length_skill, goal_state = self.select_skill()

			self.set_state(starting_state)
			self.set_goal_state(goal_state)

			self.max_steps = length_skill

			self.rollout_steps = 0
			self.traj_gripper = []
			self.traj_object = []

			state = copy.deepcopy(self.env.get_state())
			gripper_pos = self.get_gripper_pos(state)
			object_pos = self.get_object_pos(state) # best way to access object position so far

			self.traj_gripper.append(gripper_pos)
			self.traj_object.append(object_pos)

			return OrderedDict([
					("observation", state.copy()),
					("achieved_goal", self.project_to_goal_space(state).copy()),
					("desired_goal", self.goal.copy()),])
