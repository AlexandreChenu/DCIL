#!/usr/bin python -w

import datetime
import os
from os import path
import array
import time
import random
import pickle
import copy
import argparse
import gym
import matplotlib.pyplot as plt

import numpy as np

class DemoExtractor():
	"""
	Object in charge of preparing the demonstration for GGI
	"""

	def __init__(self, path, demo_filename, env_name = "FetchEnv", env_args = {}, eps_state = 0.2,  verbose = 1):
		self.path = path
		self.demo_filename = demo_filename
		self.env_name = str(env_name)
		self.env_args = env_args
		self.env_option = self.env_args["env_option"]


		self.verbose = verbose

		self.eps_state = eps_state ## threshold used for cleaning the trajectory and creating tasks

		self.incl_extra_full_state = False

		self.m_goals = None
		self.std_goals = None

		self.width_reward = 0.05

		self.L_states, self.L_inner_states, self.L_actions = self.extract_from_demo()
		self.L_states = self.L_states[1:]
		## remove extra dimensions
		self.L_states = [state[:268] for state in self.L_states]
		self.L_inner_states = self.L_inner_states[:-1]
		## remove extra dimensions
		for i in range(len(self.L_inner_states)):
			self.L_inner_states[i] = (self.L_inner_states[i][0],
										self.L_inner_states[i][1],
										self.L_inner_states[i][2],
										self.L_inner_states[i][3][:268],
										self.L_inner_states[i][4],
										self.L_inner_states[i][5],
										self.L_inner_states[i][6],
										self.L_inner_states[i][7],
										self.L_inner_states[i][8])

		# self.L_states = self.L_states[40:70]
		# self.L_inner_states = self.L_inner_states[40:70]
		# self.L_actions = self.L_actions[40:70]

		print("len(self.L_states) = ", len(self.L_states))
		self.L_full_demonstration = copy.deepcopy(self.L_states)
		self.L_full_inner_demonstration = copy.deepcopy(self.L_inner_states)
		self.L_states, self.L_inner_states, self.starting_states, self.starting_inner_states, self.L_full_observations, self.L_actions, self.L_budgets = self.clean_demonstration_trajectory()

		assert len(self.L_actions) == len(self.L_full_observations)

		## reduce to grasping only
		# self.L_states = self.L_states[8:14]
		# self.L_inner_states = self.L_inner_states[8:14]
		# self.L_budgets = self.L_budgets[8:14]

		### artificially increase the budget for the grasping task in order to facilitate manoeuvres
		# self.L_budgets[5] = self.L_budgets[5]*4

		self.L_goals = [self.project_to_goal_space(state) for state in self.L_states]

		# print("self.L_goals = ", self.L_goals)


	def extract_from_demo(self):
		"""
		Extract data from .demo files (FetchEnv & AntMazeEnv).

		Return:
			- a list of MjSimData objects corresponding to inner states
			- a list of array corresponding to full states
			- a list of actions (for BC loss)
		"""
		L_inner_states = []
		L_states = []
		L_actions = []

		if self.verbose:
			print("filename :\n", self.demo_filename)

		if not os.path.isfile(self.demo_filename):
			print ("File does not exist.")
			return

		with open(self.demo_filename, "rb") as f:
			demo = pickle.load(f)

		for i in range(len(demo["checkpoints"])):
			L_inner_states.append(demo["checkpoints"][i])

		for state, action in zip(demo["obs"], demo["actions"]):
			L_states.append(state)
			L_actions.append(action)

		return L_states, L_inner_states, L_actions

	# def change_euler2quat(self):
	# 	"""
	# 	Change angles for euler to quaternions in demo
	# 	"""
	# 	new_L_states = []
	#
	# 	for state in  self.L_states:
	# 		new_state = []
	# 		for i in range(0,18):
	# 			euler = state[i*3:(i+1)*3]
	# 			# print("euler = ", euler)
	# 			quat = gym.envs.robotics.rotations.euler2quat(euler)
	# 			# print("quat = ", quat)
	# 			new_state += list(quat.flatten())
	#
	# 		new_state = new_state + list(state[18*3:])
	# 		new_state = np.array(new_state[:604])
	# 		new_L_states.append(new_state)
	#
	# 	self.L_states = new_L_states
	#
	# 	new_L_inner_states = []
	# 	for inner_state, state in zip(self.L_inner_states, self.L_states):
	# 		new_inner_state = [inner_state[i] for i in range(len(inner_state))]
	# 		new_inner_state[3] = copy.deepcopy(state)
	# 		new_L_inner_states.append((new_inner_state))
	#
	# 	self.L_inner_states = new_L_inner_states
	#
	# 	return


	def clean_demonstration_trajectory(self, return_action=True):
		"""
		Clean the demonstration trajectory according to hyperparameter eps_dist

		L_full observations is a list of extra observations for data augmentation
		to improve the BC loss. Observations for task n are duplicated with goals
		from tasks n+1, n+2, ...
		"""
		clean_states = []
		clean_inner_states = []
		starting_states = []
		starting_inner_states = []
		L_full_observations = []
		L_actions = []

		## init lists
		clean_states.append(self.L_states[0])
		clean_inner_states.append(self.L_inner_states[0])
		# starting_states.append([self.L_states[0]])
		# starting_inner_states.append([self.L_inner_states[0]])

		init_full_obs = {"observation":copy.deepcopy(clean_states[0]),
					"achieved_goal":self.project_to_goal_space(clean_states[0]).copy(),
					"desired_goal":None}
		L_full_observations.append(init_full_obs)
		if return_action:
			L_actions.append(self.L_actions[0])

		i = 0
		L_budgets = []

		# print("len(self.L_inner_states) = ", len(self.L_inner_states))
		# print("len(self.L_states) = ", len(self.L_states))
		#
		# print("self.L_states[0] = ", self.L_states[0][:15])
		# print("self.L_states[1] = ", self.L_states[1][:15])
		#
		# # for indx in range(len(self.L_inner_states[0])):
		# # 	print("i = ", indx)
		# # 	print(self.L_inner_states[0][indx])
		# print("self.L_inner_states[0] = ", self.L_inner_states[0][3][:15])
		# print("self.L_inner_states[1] = ", self.L_inner_states[1][3][:15])

		while i < len(self.L_states)-1:
			k = 1

			sum_dist = 0

			starting_state_set = [self.L_states[i]]
			starting_inner_state_set = [self.L_inner_states[i]]
			# cumulative distance
			while sum_dist <= self.eps_state and i + k < len(self.L_states) - 1:
				sum_dist += self.compute_distance_in_goal_space(self.project_to_goal_space(self.L_states[i+k]), self.project_to_goal_space(self.L_states[i+k-1]))

				# starting_state_set.append(self.L_states[i+k])
				# starting_inner_state_set.append(self.L_inner_states[i+k])

				k += 1

				## save full observation with empty desired goal (filled later) & action
				full_obs = {"observation":copy.deepcopy(self.L_states[i+k]),
							"achieved_goal":self.project_to_goal_space(self.L_states[i+k]).copy(),
							"desired_goal":None}
				L_full_observations.append(full_obs)

				if return_action:
					L_actions.append(self.L_actions[i+k]) ## add incomplete observation & associated action


			if sum_dist > self.eps_state or i + k == len(self.L_states) - 1:
				clean_states.append(self.L_states[i+k])
				clean_inner_states.append(self.L_inner_states[i+k])
				starting_states.append(starting_state_set)
				starting_inner_states.append(starting_inner_state_set)

			# distance threshold
			# while self.compute_distance_in_goal_space(self.project_to_goal_space(self.L_states[i], default = False), self.project_to_goal_space(self.L_states[i+k], default = False)) <= self.eps_state and i + k < len(self.L_states)-1:
			# 	#print("dist = ", np.linalg.norm( self.project_to_goal_space(self.L_states[i+k]) - self.project_to_goal_space(self.L_states[i])))
			# 	full_obs = {"observation":copy.deepcopy(self.L_states[i]),
			# 				"achieved_goal":self.project_to_goal_space(self.L_states[i]).copy(),
			# 				"desired_goal":None}
			# 	L_full_observations.append(full_obs)
			# 	if return_action:
			# 		L_actions.append(self.L_actions[i]) ## add incomplete observation & associated action
			#
			# 	k += 1
			#
			# clean_states.append(self.L_states[i+k])
			# clean_inner_states.append(self.L_inner_states[i+k-1])

			## fill full observations with corresponding task goal
			add_full_observations = []
			add_actions = []
			for j in range(0, len(L_full_observations)):
				if L_full_observations[j]["desired_goal"] is None:
					new_full_obs = {"observation":copy.deepcopy(L_full_observations[j]["observation"]),
								"achieved_goal":copy.deepcopy(L_full_observations[j]["achieved_goal"]),
								"desired_goal":self.project_to_goal_space(clean_states[-1])}
					add_full_observations.append(new_full_obs)
					if return_action:
						add_actions.append(L_actions[j]) ## add completed observation & associated action but keep incomplete observation for data augmentation

			L_full_observations += add_full_observations
			L_actions += add_actions


			L_budgets.append(int(k*2)) ## add extra budget for better exploration

			## keep an extra incomplete observation for data augmentation with next tasks goals
			full_obs = {"observation":copy.deepcopy(self.L_states[i]),
						"achieved_goal":self.project_to_goal_space(self.L_states[i]).copy(),
						"desired_goal":None}
			L_full_observations.append(full_obs)
			if return_action:
				L_actions.append(self.L_actions[i])

			i = i + k

		### data augmentation for BC
		for j in range(0, len(L_full_observations)):
			if L_full_observations[j]["desired_goal"] is None:
				L_full_observations[j]["desired_goal"] = self.project_to_goal_space(clean_states[-1]) ## complete observations with last desired goal

		return clean_states, clean_inner_states, starting_states, starting_inner_states, L_full_observations, L_actions, L_budgets


	def get_env(self):
		"""
		Create environment
		"""
		print("self.env_name = ", self.env_name)
		if "FetchEnv" in self.env_name:
			env = gym.make(self.env_name, L_full_demonstration = self.L_full_demonstration,
										  L_full_inner_demonstration = self.L_full_inner_demonstration,
										  L_states = self.L_states,
										  starting_states = self.starting_states,
										  starting_inner_states = self.starting_inner_states,
										  L_actions = self.L_actions,
										  L_full_observations = self.L_full_observations,
										  L_goals = self.L_goals,
										  L_inner_states = self.L_inner_states,
										  L_budgets = self.L_budgets,  env_option = self.env_option, do_overshoot = self.env_args["do_overshoot"])

		return env

	def check_grasping(self, state):
		"""
		Check if the object is grasped in the case of Fetch environment
		"""
		collision_l_gripper_link_obj = state[216 + 167]
		collision_r_gripper_link_obj = state[216 + 193]
		# collision_l_gripper_link_obj = state[234 + 167]
		# collision_r_gripper_link_obj = state[234 + 193]

		if collision_l_gripper_link_obj and collision_r_gripper_link_obj :
			grasping = 1
		else:
			grasping = 0

		return grasping

	def project_to_goal_space(self, state, default = False):
		"""
		Project a state in the goal space depending on the environment.
		"""

		if "Fetch" in self.env_name:

			assert len(list(state))== 268 + 336 * self.incl_extra_full_state

			## position of gripper
			gripper_pos = state[84:87]
			# gripper_pos = state[102:105]
			assert len(list(gripper_pos))== 3

			## position of object
			object_pos = state[105:108]
			# object_pos = state[123:126]
			assert len(list(object_pos))==3

			## position velocity of gripper
			gripper_velp = state[138:141]
			# gripper_velp = state[156:159]
			assert len(list(gripper_velp))==3

			## gripper orientation
			gripper_quat = state[36:40]
			assert len(list(gripper_quat))==4

			## gripper orientation (euler angles)
			gripper_euler = state[27:30]
			assert len(list(gripper_euler))==3

			norm_gripper_velp = np.linalg.norm(gripper_velp)

			if default:
				return np.array(gripper_pos)

			elif "full" in self.env_option:
				return np.concatenate((np.array(gripper_pos), np.array(object_pos)))

			elif "grasping" in self.env_option:
				bool_grasping = self.check_grasping(state)
				return np.concatenate((np.array(gripper_pos), np.array([int(bool_grasping)])))

			else:
				return np.array(gripper_pos)

		else:
			print("ERROR: unknown environment")
			return 0



	def compute_distance_in_goal_space(self, goal1, goal2):
		"""
		Compute the distance in the goal space between two goals.
		"""
		if "Fetch" in self.env_name:
			if "grasping" in self.env_option:
				if len(goal1.shape) ==  1:
					euclidian_goal1 = goal1[:3]
					euclidian_goal2 = goal2[:3]

					if goal1[3] == goal2[3]:
						return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)

					else:
						return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1) #+ 1000000

				else:
					euclidian_goal1 = goal1[:,:3]
					euclidian_goal2 = goal2[:,:3]

					goal1_bool = goal1[:,3]
					goal2_bool = goal2[:,3]

					grasping_penalty = ((goal1_bool == goal2_bool).astype(np.float32)-1)*(-1000000)

					# return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1) + grasping_penalty
					return np.linalg.norm(euclidian_goal1 - euclidian_goal2, axis=-1)

			elif "full" in self.env_option:
				if len(goal1.shape) ==  1:
					return np.linalg.norm(goal1 - goal2, axis=-1)
				else:
					return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)

		else:
			if len(goal1.shape) ==  1:
				return np.linalg.norm(goal1 - goal2, axis=-1)
			else:
				return np.linalg.norm(goal1[:,:] - goal2[:,:], axis=-1)

	def visu_demonstration(self):

		if "FetchEnv" in self.env_name:
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')

			## scatter plot demo
			demo = self.L_states
			X_demo = [self.project_to_goal_space(state, default = True)[0] for state in demo]
			Y_demo = [self.project_to_goal_space(state, default = True)[1] for state in demo]
			Z_demo = [self.project_to_goal_space(state, default = True)[2] for state in demo]
			ax.scatter(X_demo, Y_demo, Z_demo, color = "blue", alpha = 0.8)

			if "full" in self.env_option:
				X_demo_object = [self.project_to_goal_space(state)[3] for state in demo]
				Y_demo_object = [self.project_to_goal_space(state)[4] for state in demo]
				Z_demo_object = [self.project_to_goal_space(state)[5] for state in demo]
				ax.scatter(X_demo_object, Y_demo_object, Z_demo_object, color = "gray", alpha = 0.8)

			full_demo = self.L_full_demonstration
			X_demo = [self.project_to_goal_space(state, default = True)[0] for state in full_demo]
			Y_demo = [self.project_to_goal_space(state, default = True)[1] for state in full_demo]
			Z_demo = [self.project_to_goal_space(state, default = True)[2] for state in full_demo]
			ax.plot(X_demo, Y_demo, Z_demo, color = "blue", alpha = 0.8)

			if "full" in self.env_option:
				X_demo_object = [self.project_to_goal_space(state)[3] for state in demo]
				Y_demo_object = [self.project_to_goal_space(state)[4] for state in demo]
				Z_demo_object = [self.project_to_goal_space(state)[5] for state in demo]
				ax.scatter(X_demo_object, Y_demo_object, Z_demo_object, color = "gray", alpha = 0.8)

			for azim_ in range(45,360,90):
				ax.view_init(azim = azim_)
				plt.savefig(self.path + "/demonstration_" + str(azim_) + ".png")
			plt.close(fig)


		return
