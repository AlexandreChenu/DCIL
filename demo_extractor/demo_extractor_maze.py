#!/usr/bin python -w

import datetime
import os
from os import path
import array
import time
import random
import pickle5 as pickle
import copy
import argparse
import gym
import matplotlib.pyplot as plt

import numpy as np

class DemoExtractor():
	"""
	Object in charge of extracting skills from the expert demonstration
	"""

	def __init__(self, path,
					demo_filename,
					env_name = "MazeEnv",
					env_args = {},
					eps_state = 0.2,
					width_reward = 0.1,
					verbose = 1):

		self.path = path
		self.demo_filename = demo_filename
		self.env_name = str(env_name)
		self.env_args = env_args
		# self.env = gym.make(self.env_name)

		self.verbose = verbose

		self.eps_state = eps_state ## threshold used for cleaning the trajectory and creating tasks
		self.beta = 1.25 ## extra control steps for skill learning

		self.incl_extra_full_state = True

		self.m_goals = None
		self.std_goals = None

		self.width_reward = 0.1

		## raw extraction of states, inner_state (states for simulation reset in Mujoco), actions
		self.L_states, self.L_inner_states = self.extract_from_demo()
		self.L_full_demonstration = copy.deepcopy(self.L_states) ## (kept for visualization)

		## divide in sub-trajectories and extract first state
		self.L_states, self.L_inner_states, self.L_budgets = self.clean_demonstration_trajectory()

		## list of tasks goals
		self.L_goals = [self.project_to_goal_space(state) for state in self.L_states]

	def extract_from_demo(self):
		"""
		Extract states, mujoco formated states from .demo files.

		Return:
			- a list of MjSimData objects corresponding to inner states
			- a list of array corresponding to full states
		"""
		L_inner_states = []
		L_states = []

		if self.verbose:
			print("filename :\n", self.demo_filename)

		if not os.path.isfile(self.demo_filename):
			print ("File does not exist.")
			return

		with open(self.demo_filename, "rb") as f:
			demo = pickle.load(f)

		for state, action in zip(demo["obs"], demo["actions"]):
			L_states.append(state)
			L_inner_states.append(state)

		return L_states, L_inner_states

	def clean_demonstration_trajectory(self):
		"""
		Clean the demonstration trajectory according to hyperparameter eps_dist

		L_full observations is a list of extra observations for data augmentation
		to improve the BC loss. Observations for task n are duplicated with goals
		from tasks n+1, n+2, ...
		"""
		clean_states = []
		clean_inner_states = []
		L_full_observations = []

		## init lists
		clean_states.append(self.L_states[0])
		clean_inner_states.append(self.L_inner_states[0])

		i = 0
		L_budgets = []

		while i < len(self.L_states)-1:
			k = 1

			sum_dist = 0

			# cumulative distance
			while sum_dist <= self.eps_state and i + k < len(self.L_states) - 1:
				sum_dist += self.compute_distance_in_goal_space(self.project_to_goal_space(self.L_states[i+k]), self.project_to_goal_space(self.L_states[i+k-1]))
				k += 1

			if sum_dist > self.eps_state or i + k == len(self.L_states) - 1:
				clean_states.append(self.L_states[i+k])
				clean_inner_states.append(self.L_inner_states[i+k])

			L_budgets.append(int(self.beta*k))

			i = i + k

		return clean_states, clean_inner_states, L_budgets


	def get_env(self):
		"""
		Create environment including skill manager
		"""
		print("self.env_name = ", self.env_name)

		if "MazeEnv" in self.env_name:
			print("self.env_name = ", self.env_name)
			env = gym.make(self.env_name, L_full_demonstration = self.L_full_demonstration,
										  L_states = self.L_states,
										  L_goals = self.L_goals,
										  L_inner_states = self.L_inner_states,
										  L_budgets = self.L_budgets, mazesize = self.env_args["mazesize"], do_overshoot = self.env_args["do_overshoot"])

		return env

	def project_to_goal_space(self, state, default = False):
		"""
		Project a state in the goal space depending on the environment.
		"""

		if "Maze" in self.env_name:
			return np.array(state[:2])

		else:
			print("ERROR: unknown environment")
			return 0



	def compute_distance_in_goal_space(self, goal1, goal2):
		"""
		Compute the distance in the goal space between two goals.
		"""

		if "Maze" in self.env_name:
			return np.linalg.norm(goal1 - goal2, axis=-1)

		else:
			if len(goal1.shape) ==  1:
				return np.linalg.norm(goal1 - goal2, axis=-1)
			else:
				return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)

	def visu_demonstration(self):
		"""
		Plot the demonstration extracted
		"""

		if "MazeEnv" in self.env_name:
			fig = plt.figure()
			ax = fig.add_subplot()

			## scatter plot demo
			demo = self.L_states
			X_demo = [self.project_to_goal_space(state)[0] for state in demo]
			Y_demo = [self.project_to_goal_space(state)[1] for state in demo]
			ax.scatter(X_demo, Y_demo, color = "blue", alpha = 0.8)

			full_demo = self.L_full_demonstration
			X_demo = [self.project_to_goal_space(state)[0] for state in full_demo]
			Y_demo = [self.project_to_goal_space(state)[1] for state in full_demo]
			ax.plot(X_demo, Y_demo, color = "blue", alpha = 0.8)

			plt.savefig(self.path + "/demonstration.png")
			plt.close(fig)

		else:
			print("Unknown env")
		return
