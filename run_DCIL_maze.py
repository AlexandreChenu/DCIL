#!/usr/bin python -w

import datetime
import os
from os import path
import array
import time
import random
import pickle
# import pickle5 as pickle
import copy
import argparse
import pdb

from datetime import datetime

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


from stable_baselines3 import HerReplayBuffer#, PPO, DDPG, TD3#, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.buffers import DictRolloutBuffer

from algos.SAC_DCIL import SAC
from algos.TQC_DCIL import TQC
from demo_extractor.demo_extractor_maze import DemoExtractor
from evaluate.dubinsmazeenv.evaluate_mazeenv import eval_trajectory_mazeenv
from callbacks.callbacks import LogCallbackMazeEnv

from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print("device = ", device)

def learn_DCIL(args, env, eval_env, path):
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
                                        n_sampled_goal=4,
                                        goal_selection_strategy=goal_selection_strategy,
                                        online_sampling=online_sampling,
                                        max_episode_length=max_episode_length,
                                        ),
                                        ent_coef=0.1,
                                        policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300]),
                                        warmup_duration=100,
                                        verbose=1, path=path, make_logs = False,
                                        bonus_reward_bool = args["bonus_reward_bool"],
                                        add_ent_reg_critic = args["add_ent_reg"],
                                        alpha_bonus = 1.,
                                        device= device)

    if args["RL_algo"] == "TQC_HER":
        # Available strategies (cf paper): future, final, episode
        goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
        # If True the HER transitions will get sampled online
        online_sampling = True

        # Time limit for the episodes
        max_episode_length = 50
        ##### Warning: should it be fixed or can it be variable

        model = TQC("MultiInputPolicy", env,
                                        learning_rate = 1e-3,
                                        gamma = 0.99,
                                        batch_size = 256,
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
                                        policy_kwargs = dict(log_std_init=-3, net_arch=[400,300]),
                                        #policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300], optimizer_class=torch.optim.RMSprop, optimizer_kwargs=dict(eps=args["eps_optimizer"])),
                                        verbose=1,
                                        device= device,
                                        path=path, make_logs = False,
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
    f_ratio = open(path + "/ratio.txt", "w")
    f_nb_skill_succeeded = open(path + "/nb_skill_succeeded.txt", "w")
    f_max_zone= open(path + "/max_zone.txt", "w")

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
            f_ratio.write(str(ratio) + "\n")
            print("|    success ratio (successful rollouts / total rollouts) =  ", ratio)

            ## evaluate chaining of skills
            eval_traj, skills_successes, max_zone = eval_trajectory_mazeenv(env, eval_env, model, args["algo_type"])
            print("|    skill-chaining: ", skills_successes)
            successfull_traj = skills_successes[-1]
            print("|    skill-chaining success: ", successful_traj)
            print("------------------------------------------------------------------------------------------------------------")


            f_nb_skill_succeeded.write(str(sum([int(skill_success) for skill_success in skills_successes])) + "\n")
            f_max_zone.write(str(max_zone) + "\n")

            ## stop training if successful chaining of skill
            if successful_traj:
                callback.callbacks[0].eval_trajs.append(eval_traj)
                callback.callbacks[0]._visu_trajectories(env, eval_env,
                                                        callback.callbacks[0].eval_trajs,
                                                        callback.callbacks[0].trajs, callback.callbacks[0].success_trajs)
                break

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

    f_ratio.close()
    f_nb_skill_succeeded.close()
    f_max_zone.close()

    total_nb_timesteps = model.num_timesteps
    callback.on_training_end()

    return total_nb_timesteps

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument for GCP imitation.')
    parser.add_argument('--eps_dist', help='espilon distance for cleaning trajectory')
    parser.add_argument('--algo', help='RL algorithm')
    parser.add_argument('--env', help='environment')
    parser.add_argument('--demo_path', help='demostration file')
    parser.add_argument('--size', help='maze size')
    parser.add_argument('--bonus_bool', help='add bonus reward')
    parser.add_argument('--overshoot_bool', help='overshoot if success yes(1) no(0)')
    parser.add_argument('--add_ent_reg', help='add entropy regularization for critic')
    parser.add_argument('-x', help='demo indx')

    parsed_args = parser.parse_args()

    args = {}
    args["eps_dist"] = float(parsed_args.eps_dist)
    args["success_dist"] = 0.2
    args["success_ratio"] = 0.98
    args["num_episodes"] = 100
    args["episode_timesteps"] = 200
    args["RL_algo"] = str(parsed_args.algo)
    args["env_name"] = str(parsed_args.env)
    args["mazesize"] = str(parsed_args.size)
    args["demo_directory"] = str(parsed_args.demo_path)
    args["demo_indx"] = int(parsed_args.x)
    args["bonus_reward_bool"] = bool(int(parsed_args.bonus_bool))
    args["do_overshoot"] = bool(int(parsed_args.overshoot_bool))
    args["add_ent_reg"] = bool(int(parsed_args.add_ent_reg))
    # args["demo_filename"] = args["demo_directory"] + "/" + str(np.random.randint(1,21)) + ".demo"
    args["demo_filename"] = args["demo_directory"] + "/" + str(args["demo_indx"]) + ".demo"
    args["total_timesteps"] = 200000


    if "DDPG" in args["RL_algo"] or "SAC" in args["RL_algo"] or "TD3" in args["RL_algo"]:
        args["algo_type"] = "OffPolicyAlgorithm"
    else:
        args["algo_type"] = "OnPolicyAlgorithm"


    # create new directory
    now = datetime.now()
    dt_string = '_%s_%s' % (datetime.now().strftime('%Y%m%d'), str(os.getpid()))

    cur_path = os.getcwd()
    dir_path = cur_path + "/xp/DCIL_" + args["env_name"] + "_" + args["RL_algo"] + "_" + str(args["bonus_reward_bool"]) + "_" + str(args["do_overshoot"]) + "_" + str(args["demo_indx"]) + "_" + dt_string

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

    # pdb.set_trace()

    ## extract demonstration from demo file
    demo_extractor = DemoExtractor(dir_path,
                                    args["demo_filename"],
                                    env_name = env_name,
                                    env_args = {"mazesize":args["mazesize"], "do_overshoot":args["do_overshoot"]},
                                    eps_state = args["eps_dist"])
    demo_extractor.visu_demonstration()

    ## create environment
    env = demo_extractor.get_env()
    assert len(env.skill_manager.L_inner_states) == len(env.skill_manager.L_states)
    assert len(env.skill_manager.starting_state_set) == len(env.skill_manager.L_states)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=np.inf)
    env = VecMonitor(env)

    ## create eval environment
    eval_env = demo_extractor.get_env()
    assert len(eval_env.skill_manager.L_inner_states) == len(eval_env.skill_manager.L_states)
    assert len(eval_env.skill_manager.starting_state_set) == len(eval_env.skill_manager.L_states)

    random.seed(None)

    # learn GCP
    total_nb_timesteps = learn_DCIL(args, env, eval_env, dir_path)

    f = open(dir_path + "/nb_timesteps.txt", "a")
    f.write(str(total_nb_timesteps))
    f.close()
