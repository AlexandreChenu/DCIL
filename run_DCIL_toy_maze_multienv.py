#!/usr/bin python -w
import datetime
from datetime import datetime
import os
from os import path
import array
import time
import random
import pickle
import copy
import argparse
import pdb

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as mc

import gym
from envs import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data


from stable_baselines3 import HerReplayBuffer, SAC # PPO, DDPG, TD3#, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.buffers import DictRolloutBuffer

# from algos.SAC_DCIL import SAC
# from algos.TQC_DCIL import TQC
from demo_extractor.demo_extractor_maze import DemoExtractor
from evaluate.dubinsmazeenv.evaluate_mazeenv import eval_trajectory_mazeenv
from callbacks.callbacks import LogCallbackMazeEnv

from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from stable_baselines3.common.env_util import make_vec_env

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")



def learn_GGI(args, env, eval_env, path):
	"""
	Learn goal-conditioned policy for Goal Guided Imitation
	"""
	## init RL algorithm
	if args["RL_algo"] == "SAC_HER":
		# Available strategies (cf paper): future, final, episode
		goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
		# If True the HER transitions will get sampled online
		online_sampling = True

		# Time limit for the episodes
		max_episode_length = 50
		##### Warning: should it be fixed or can it be variable

		model = SAC("MultiInputPolicy", env, #eval_env.L_states, eval_env.L_steps,
					learning_rate = 1e-3, replay_buffer_class=HerReplayBuffer,
					# Parameters for HER
					replay_buffer_kwargs=dict(
					n_sampled_goal=2,
					goal_selection_strategy=goal_selection_strategy,
					online_sampling=online_sampling,
					# max_episode_length=max_episode_length
					),
					ent_coef= 0.1,
					policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300]),
					train_freq= 1,
					gradient_steps = env.num_envs,
					# warmup_duration=100,
					verbose=1,path=path,# make_logs = False,
					eval_env = eval_env,
					add_bonus_reward = args["bonus_reward_bool"],
					# add_ent_reg_critic = args["add_ent_reg"],
					# alpha_bonus = 0.1,
					device= device)

	if args["RL_algo"] == "TQC_HER":
		# Available strategies (cf paper): future, final, episode
		goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
		# If True the HER transitions will get sampled online
		online_sampling = True

		# Time limit for the episodes
		max_episode_length = 50
		##### Warning: should it be fixed or can it be variable

		# Add action noise for "more i.i.d" transitions?
		# n_actions = env.action_space.shape[-1]
		# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

		model = TQC("MultiInputPolicy", env,
										learning_rate = 1e-3,
										gamma = 0.99,
										batch_size = 1024,
										learning_starts = 100,
										replay_buffer_class=HerReplayBuffer,
										# Parameters for HER
										replay_buffer_kwargs=dict(
										n_sampled_goal=4,
										goal_selection_strategy=goal_selection_strategy,
										online_sampling=online_sampling,
										max_episode_length=max_episode_length,
										),
										ent_coef=0.1,
										policy_kwargs = dict(log_std_init=-3, net_arch=[512, 512, 512]),
										#policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300], optimizer_class=torch.optim.RMSprop, optimizer_kwargs=dict(eps=args["eps_optimizer"])),
										verbose=1,
										device= device,
										add_bonus_reward = args["bonus_reward_bool"],
										add_ent_reg_critic = args["add_ent_reg"])

	## setup callback and learning
	callback = LogCallbackMazeEnv(path, args["RL_algo"], args["algo_type"], eval_env)
	total_timesteps, callback = model._setup_learn(args["total_timesteps"],
														eval_env, callback,
														-1, 5, args["RL_algo"],
														True, args["algo_type"])

	## opening some log files
	# f_m_r = open(path + "/m_r.txt", "w")
	# f_m_l = open(path + "/m_l.txt", "w")
	# f_ratio = open(path + "/ratio.txt", "w")
	f_nb_chained_skills = open(path + "/nb_chained_skills.txt", "w")

	callback.on_training_start(locals(), globals())

	rollout_collection_cnt = 0
	successful_traj = False
	last_episode_num = 0

	############################################################################
	########################
	########################            Training loop
	########################
	############################################################################

	while model.num_timesteps < total_timesteps:

		rollout_collection_cnt += 1

		if args["algo_type"] == "OnPolicyAlgorithm":
			continue_training = model.collect_rollouts(model.env,
														callback,
														model.rollout_buffer,
														n_rollout_steps=model.n_steps)

		if args["algo_type"] == "OffPolicyAlgorithm":
			rollout = model.collect_rollouts(model.env, train_freq = model.train_freq,
														action_noise = model.action_noise,
														callback = callback,
														learning_starts = model.learning_starts,
														replay_buffer = model.replay_buffer,
														log_interval = 10)


			continue_training = rollout.continue_training

		if continue_training is False:
			break

		if rollout_collection_cnt > 100:
			print("------------------------------------------------------------------------------------------------------------")
			print("| skills/")
			print("|    nb of successfull skill-rollouts: ", callback.callbacks[0].sum_W)
			print("|    total nb of skill-rollouts = ", callback.callbacks[0].n_runs)
			sum_W = callback.callbacks[0].sum_W
			n_runs = callback.callbacks[0].n_runs

			callback.callbacks[0].sum_W = 0
			callback.callbacks[0].n_runs = 0

			if float(n_runs) > 0.:
				ratio = float(sum_W) / float(n_runs)
			else:
				ratio = 0.
			print("|    success ratio (successful rollouts / total rollouts) =  ", ratio)

			## evaluate chaining of skills
			eval_traj, skills_successes, max_zone = eval_trajectory_mazeenv(env, eval_env, model, args["algo_type"])
			print("|    skill-chaining: ", skills_successes)
			successfull_traj = skills_successes[-1]
			print("|    skill-chaining success: ", successful_traj)
			print("------------------------------------------------------------------------------------------------------------")

			# f_ratio.write(str(ratio) + "\n")
			f_nb_chained_skills.write(str(sum([int(skill_success) for skill_success in skills_successes])) + "\n")

			rollout_collection_cnt = 0

		### train model
		if args["algo_type"] == "OnPolicyAlgorithm":
			model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
			model.train()

		if args["algo_type"] == "OffPolicyAlgorithm":
			if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
				# If no `gradient_steps` is specified,
				# do as many gradients steps as steps performed during the rollout
				gradient_steps = model.gradient_steps if model.gradient_steps > 0 else rollout.episode_timesteps
				model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)

	# f_ratio.close()
	f_nb_chained_skills.close()

	total_nb_timesteps = model.num_timesteps
	callback.on_training_end()

	## close all file handlers
	if model.make_logs:
		model.f_log_next_q_values.close()
		model.f_log_next_q_values = None
		model.f_log_target_q_values.close()
		model.f_log_target_q_values = None
		model.f_next_log_prob.close()
		model.f_next_log_prob = None
		model.f_log_current_q_values.close()
		model.f_log_current_q_values = None
		model.f_weights_sum.close()
		model.f_weights_sum = None
		model.f_critic_losses.close()
		model.f_critic_losses = None
		model.f_actor_losses.close()
		model.f_actor_losses = None

	# eval_traj, skills_successes, max_zone = eval_trajectory_mazeenv(env, eval_env, model, args["algo_type"])
	# print("eval_traj = ", eval_traj[:10])

	## save model
	model.save(path + "/GCP_model")

	# ## test model reload
	# del model
	# model = SAC.load(path + "/GCP_model", env=env, custom_objects={"path":path})
	# print("model.env = ", model.env)
	# print("model.env.envs[0].skills = ", model.env.envs[0].skills)
	# print("model.env.envs[0].skills.L_skills_feasible = ", model.env.envs[0].skills.L_skills_results)
	# eval_traj, skills_successes, max_zone = eval_trajectory_mazeenv(env, eval_env, model, args["algo_type"])
	# print("eval_traj = ", eval_traj[:10])

	return total_nb_timesteps

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Argument for GCP imitation.')
	parser.add_argument('--algo', help='RL algo')
	parser.add_argument('--add_bonus', help='add bonus reward')
	parser.add_argument('--add_ent_reg', help='add entropy regularization for critic')
	parser.add_argument('--num_envs', help='number of parallel envs')
	parsed_args = parser.parse_args()

	args = {}
	args["bonus_reward_bool"] = bool(int(parsed_args.add_bonus))
	args["add_ent_reg"] = bool(int(parsed_args.add_ent_reg))
	args["n_envs"] = int(parsed_args.num_envs)

	args["eps_dist"] = 1.2
	args["success_dist"] = 0.2
	args["success_ratio"] = 0.98
	args["num_episodes"] = 100
	args["episode_timesteps"] = 200
	# args["RL_algo"] = "SAC_HER"
	args["RL_algo"] = str(parsed_args.algo)
	args["env_name"] = "DubinsMazeEnv"
	args["mazesize"] = "2"
	args["bc_reg_bool"] = False
	args["do_overshoot"] =  True
	args["gamma"] = 0.99
	args["alpha"] = 0.1
	args["total_timesteps"] = 35000
	# args["n_envs"] = 1


	if "DDPG" in args["RL_algo"] or "SAC" in args["RL_algo"] or "TD3" in args["RL_algo"] or "TQC" in args["RL_algo"]:
		args["algo_type"] = "OffPolicyAlgorithm"
	else:
		args["algo_type"] = "OnPolicyAlgorithm"


	# create new directory
	now = datetime.now()
	dt_string = '_%s_%s' % (datetime.now().strftime('%Y%m%d'), str(os.getpid()))

	cur_path = os.getcwd()
	dir_path = cur_path + "/xp/DCIL_" + args["env_name"] + "_" + args["RL_algo"] + "_" + str(args["bonus_reward_bool"]) + "_" + str(args["do_overshoot"]) + "_" + str(args["add_ent_reg"]) + "_" + str(args["gamma"]) + "_" + str(args["alpha"]) + "_" + dt_string

	try:
		os.mkdir(dir_path)
	except OSError:
		print ("Creation of the directory %s failed" % dir_path)
	else:
		print ("Successfully created the directory %s " % dir_path)


	if "HER" in args["RL_algo"] or "GCP" in args["RL_algo"]:
		# create local environment
		env_name = args["env_name"] + "GCPHERSB3-v0"
	else:
		env_name = args["env_name"] + "GCPSB3-v0"

	## manual extraction of skills from demonstration
	demo_filename = cur_path + "/demos/toy_dubinsmazeenv/1.demo"
	with open(demo_filename, "rb") as f:
		demo = pickle.load(f)

	L_states = []
	for state in demo["obs"]:
		L_states.append(state[:3])

	L_full_demonstration = [L_states[i] for i in range(0,len(L_states),2)]
	L_full_inner_demonstration = copy.deepcopy(L_states)
	L_states = [L_states[0], L_states[int(len(L_states)/2)], L_states[-1]]
	L_goals = [state[:2] for state in L_states]
	L_inner_states = copy.deepcopy(L_states)
	L_budgets = [20,20]

	vec_env = True
	## create environment
	if vec_env == False:
		env = gym.make(env_name, L_full_demonstration = L_full_demonstration,
									  L_states =  L_states,
									  L_goals =  L_goals,
									  L_inner_states =  L_inner_states,
									  L_budgets =  L_budgets, mazesize = args["mazesize"], do_overshoot = args["do_overshoot"])

	else:
		env = make_vec_env(env_name, n_envs = args["n_envs"], env_kwargs = {"L_full_demonstration":L_full_demonstration,
																	"L_states":L_states,
																	"L_goals":L_goals,
																	"L_inner_states":L_inner_states,
																	"L_budgets":L_budgets,
																	"mazesize":args["mazesize"],
																	"do_overshoot":args["do_overshoot"]}, vec_env_cls=SubprocVecEnv)




	# env = DummyVecEnv([lambda: env])
	# env = SubprocVecEnv([lambda: env])
	env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=np.inf)
	env = VecMonitor(env)

	## create eval environment
	eval_env = gym.make(env_name, L_full_demonstration = L_full_demonstration,
								  L_states =  L_states,
								  L_goals =  L_goals,
								  L_inner_states =  L_inner_states,
								  L_budgets =  L_budgets, mazesize = args["mazesize"], do_overshoot = args["do_overshoot"])
	args["demo_length"] = len(eval_env.skill_manager.L_states)
	## random seed to current datetime
	random.seed(None)

	# learn GCP
	total_nb_timesteps = learn_GGI(args, env, eval_env, dir_path)

	f = open(dir_path + "/nb_timesteps.txt", "a")
	f.write(str(total_nb_timesteps))
	f.close()
