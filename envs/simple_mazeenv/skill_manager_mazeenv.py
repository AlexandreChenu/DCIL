import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch



class SkillsManager():

	def __init__(self, L_full_demonstration, L_states, L_inner_states, L_goals, L_budgets):

		self.L_full_demonstration = L_full_demonstration
		self.L_states = L_states
		self.L_inner_states = L_inner_states
		self.L_goals = L_goals
		self.L_budgets = L_budgets

		## set of starting states (for subgoal adaptation)
		self.starting_inner_state_set = [[inner_state] for inner_state in self.L_inner_states]
		self.starting_state_set = [[state] for state in self.L_states]

		## a list of list of results per skill
		self.L_skills_results = [[] for _ in self.L_states]
		self.L_overshoot_results = [[[]] for _ in self.L_states]

		self.skill_window = 20
		self.max_size_starting_state_set = 100

		self.weighted_sampling = False

		self.delta_step = 1
		self.dist_threshold = 0.1

		self.nb_skills = len(self.L_states)-1

		self.subgoal_adaptation = False
		self.add_noise_starting_state = False

	def get_skill(self, skill_indx):
		"""
		get starting state, length and goal associated to a given skill
		"""
		assert skill_indx > 0
		assert skill_indx < len(self.L_states)

		self.indx_start = skill_indx - self.delta_step
		self.indx_goal = skill_indx

		length_skill = sum(self.L_budgets[self.indx_start:self.indx_goal])

		starting_state, starting_inner_state = self.get_starting_state(skill_indx - self.delta_step, test=True)
		goal_state = self.get_goal_state(skill_indx)

		return starting_inner_state, length_skill, goal_state

	def next_skill(self):
		"""
		shift skill by one and get skill-goal and length.

		if no more skill -> return False signal
		"""

		self.indx_goal += 1

		if self.indx_goal < len(self.L_states) :
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
		Monitor successes for a given skill (to bias skill selection)
		"""
		self.L_skills_results[skill_indx].append(1)

		if len(self.L_skills_results[skill_indx]) > self.skill_window:
			self.L_skills_results[skill_indx].pop(0)

		return

	def add_failure(self, skill_indx):
		"""
		Monitor failures for a given skill (to bias skill selection)
		"""
		self.L_skills_results[skill_indx].append(0)

		if len(self.L_skills_results[skill_indx]) > self.skill_window:
			self.L_skills_results[skill_indx].pop(0)

		return

	def get_skill_success_rate(self, skill_indx):

		nb_skills_success = self.L_skills_results[skill_indx].count(1)
		s_r = float(nb_skills_success/len(self.L_skills_results[skill_indx]))

		## keep a small probability for successful skills to be selected in order
		## to avoid catastrophic forgetting
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

		## fitness based selection
		if self.weighted_sampling and weights_available:

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

		## uniform sampling
		else:
			i = random.randint(self.delta_step, len(self.L_states)-1)

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

		if self.subgoal_adaptation and not overshoot:
			## uniform sampling of goal state in the starting_state_set
			indx_goal_state = random.randint(0,len(self.starting_state_set[indx_goal])-1)
			return self.starting_state_set[indx_goal][indx_goal_state]

		else:
			return self.starting_state_set[indx_goal][0]


	def project_to_goal_space(self, state):
		"""
		Project a state in the goal space depending on the environment.
		"""
		return np.array(state[:2])

	def compute_distance_in_goal_space(self, goal1, goal2):

		goal1 = np.array(goal1)
		goal2 = np.array(goal2)

		if len(goal1.shape) ==  1:
			return np.linalg.norm(goal1 - goal2, axis=-1)
		else:
			return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)
