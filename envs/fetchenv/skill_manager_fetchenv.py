import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch



class SkillManager():

	def __init__(self, L_full_demonstration, L_full_inner_demonstration, L_states, starting_states, starting_inner_states, L_actions, L_full_observations, L_goals, L_inner_states, L_budgets, m_goals, std_goals, env_option):

		self.L_full_demonstration = L_full_demonstration
		self.L_full_inner_demonstration = L_full_inner_demonstration
		self.L_states = L_states
		self.L_inner_states = L_inner_states
		self.L_budgets = L_budgets
		self.L_goals = L_goals

		## set of starting state for subgoal adaptation
		self.starting_inner_state_set = starting_inner_states
		self.starting_state_set = starting_states

		## monitor skill success rates, skill feasibility, overshoot success rates and feasibility
		self.L_skills_results = [[] for _ in self.L_states]
		self.L_overshoot_results = [[[]] for _ in self.L_states] ## a list of list of results per skill
		self.L_skills_feasible = [False for _ in self.L_states]
		self.L_overshoot_feasible = [False for _ in self.L_states]

		self.skill_window = 20
		self.max_size_starting_state_set = 15

		## skill sampling strategy (weighted or uniform)
		self.weighted_sampling = True

		self.delta_step = 1
		self.dist_threshold = 0.08

		self.nb_skills = len(self.L_states)-1

		self.env_option = env_option

		self.incl_extra_full_state = 0


		self.L_goals = [self.project_to_goal_space(state) for state in self.L_states]

	def get_skill(self, skill_indx):
		"""
		get starting state, length and goal associated to a given skill
		"""
		assert skill_indx > 0
		assert skill_indx < len(self.L_states), "indx too large: skill indx = " + str(skill_indx)

		self.indx_start = skill_indx - self.delta_step
		self.indx_goal = skill_indx

		length_skill = sum(self.L_budgets[self.indx_start:self.indx_goal])

		starting_state, starting_inner_state = self.get_starting_state(skill_indx - self.delta_step, test=True)
		goal_state = self.get_goal_state(skill_indx)

		return starting_inner_state, length_skill, goal_state

	def next_skill(self):
		"""
		shift skill by one and get skill goal and skill length if possible.

		if no more skill -> return False signal
		"""

		self.indx_goal += 1

		if self.indx_goal < len(self.L_states):
			assert self.indx_goal < len(self.L_states)

			length_skill = sum(self.L_budgets[self.indx_goal - self.delta_step:self.indx_goal])

			goal_state = self.get_goal_state(self.indx_goal, overshoot = True)

			return goal_state, length_skill, True

		else:
			return None, None, False

	def add_success_overshoot(self,skill_indx):
		self.L_overshoot_feasible[skill_indx-1]=True
		return

	def add_success(self, skill_indx):
		"""
		Monitor successes for a given skill
		"""
		self.L_skills_feasible[skill_indx] = True
		self.L_skills_results[skill_indx].append(1)

		if len(self.L_skills_results[skill_indx]) > self.skill_window:
			self.L_skills_results[skill_indx].pop(0)

		return

	def add_failure(self, skill_indx):
		"""
		Monitor failues for a given skill
		"""
		self.L_skills_results[skill_indx].append(0)

		if len(self.L_skills_results[skill_indx]) > self.skill_window:
			self.L_skills_results[skill_indx].pop(0)

		return

	def get_skill_success_rate(self, skill_indx):

		nb_skills_success = self.L_skills_results[skill_indx].count(1)

		s_r = float(nb_skills_success/len(self.L_skills_results[skill_indx]))

		## on cape l'inversion
		if s_r <= 0.1:
			s_r = 10
		else:
			s_r = 1./s_r

		return s_r

	def get_skills_success_rates(self):

		L_rates = []

		for i in range(self.delta_step, len(self.L_states)):
			L_rates.append(self.get_skill_success_rate(i))

		return L_rates


	def sample_skill_indx(self):
		"""
		Sample a skill indx.

		2 cases:
			- weighted sampling of skill according to skill success rates
			- uniform sampling
		"""
		weights_available = True
		for i in range(self.delta_step,len(self.L_skills_results)):
			if len(self.L_skills_results[i]) == 0:
				weights_available = False

		if self.weighted_sampling and weights_available: ## weighted sampling

			L_rates = self.get_skills_success_rates()

			assert len(L_rates) == len(self.L_states) - self.delta_step

			## weighted sampling
			total_rate = sum(L_rates)
			pick = random.uniform(0, total_rate)

			current = 0
			for i in range(0,len(L_rates)):
				s_r = L_rates[i]
				current += s_r
				if current > pick:
					break

			i = i + self.delta_step

		else: ## uniform sampling
			i = random.randint(self.delta_step, len(self.L_states)-1)
			# i = 2

		return i

	def select_skill(self):
		"""
		Select a skill and return corresponding starting state, budget and goal
		"""
		skill_indx = self.sample_skill_indx()
		## skill indx coorespond to a goal indx
		self.indx_start = skill_indx - self.delta_step


		self.indx_goal = skill_indx
		length_skill = sum(self.L_budgets[self.indx_start:self.indx_goal])
		starting_state, starting_inner_state = self.get_starting_state(self.indx_start)

		goal_state = self.get_goal_state(self.indx_goal)
		return starting_inner_state, length_skill, goal_state

	def get_starting_state(self, indx_start, test=False):

		if test:
			indx=  0
		else:
			if len(self.starting_state_set[indx_start]) > 2:
				# indx = np.random.randint(0,len(self.starting_state_set[indx_start])-1)
				indx = np.random.randint(0,len(self.starting_state_set[indx_start]))
			else:
				indx = 0

		return self.starting_state_set[indx_start][indx], self.starting_inner_state_set[indx_start][indx]

	def get_goal_state(self, indx_goal, overshoot=False):

		return self.L_states[indx_goal]

	def check_grasping(self, state):
		"""
		Check if the object is grasped in the case of Fetch environment
		"""

		collision_l_gripper_link_obj = state[216 + 167]
		collision_r_gripper_link_obj = state[216 + 193]
		collision_object_table = state[216 + 67] ## add collision between object and table to improve grasing check

		## quaternion angles
		# collision_l_gripper_link_obj = state[234 + 167]
		# collision_r_gripper_link_obj = state[234 + 193]
		# collision_object_table = state[234 + 67] ## add collision between object and table to improve grasing check

		if collision_l_gripper_link_obj and collision_r_gripper_link_obj and not collision_object_table :
			grasping = 1
		else:
			grasping = 0

		return grasping

	def project_to_goal_space(self, state):
		"""
		Project a state in the goal space depending on the environment.
		"""

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


	def compute_distance_in_goal_space(self, in_goal1, in_goal2):

		goal1 = copy.deepcopy(in_goal1)
		goal2 = copy.deepcopy(in_goal2)

		if "grasping" in self.env_option:

			goal1[:-1] = self._normalize_goal(goal1[:-1])
			goal2[:-1] = self._normalize_goal(goal2[:-1])

			if len(goal1.shape) ==  1:
				euclidian_goal1 = goal1[:3]
				euclidian_goal2 = goal2[:3]

				if goal1[3] == goal2[3]:
					return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)

				elif goal2[3] == 0 and goal1[3] == 1:
					return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)

				else:
					return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1) + 1000000 ## if no grasping, artificially push goals away
					#return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)
			else:
				euclidian_goal1 = goal1[:,:3]
				euclidian_goal2 = goal2[:,:3]

				goal1_bool = goal1[:,3]
				goal2_bool = goal2[:,3]

				grasping_penalty = ((goal1_bool == goal2_bool).astype(np.float32)-1)*(-1000000)

				return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1) + grasping_penalty
				#return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)

		else:
			if len(goal1.shape) ==  1:
				return np.linalg.norm(goal1 - goal2, axis=-1)
			else:
				return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)

	def get_gripper_pos(self, state):
		"""
		get gripper position from full state
		"""
		#print("len(state) = ", len(state))
		#print("state = ", state)
		assert len(list(state))== 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_pos = state[84:87]
		# gripper_pos = state[102:105]
		assert len(list(gripper_pos)) == 3

		return gripper_pos

	def get_gripper_quat(self, state):
		"""
		get object orientation from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_quat = state[36:40]
		assert len(list(gripper_quat))==4

		return gripper_quat

	def get_object_pos(self, state):
		"""
		get object position from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		object_pos = state[105:108]
		# object_pos = state[123:126]
		assert len(list(object_pos))==3

		return object_pos


	def get_gripper_euler(self, state):
		"""
		get object orientation from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_euler = state[27:30] #TODO
		assert len(list(gripper_euler))==3

		return gripper_euler

	def get_gripper_velp(self, state):
		"""
		get object position from full state
		"""
		assert len(list(state)) == 268 + 336 * self.incl_extra_full_state or len(list(state))== 268

		gripper_velp = state[138:141]
		# gripper_velp = state[156:159]
		assert len(list(gripper_velp))==3

		return gripper_velp
