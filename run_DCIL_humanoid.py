#!/usr/bin python -w

import datetime
import os, psutil
from os import path
import array
import time
import random
import pickle
import copy
import argparse

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print("device = ", device)

import cv2

torch.set_default_tensor_type('torch.FloatTensor')

from envs import *

from stable_baselines3 import HerReplayBuffer, PPO, DDPG
# from stable_baselines3 import PPO, DDPG
# from SAC_utils.her_replay_buffer import HerReplayBuffer

from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.logger import configure

from SAC_utils.SAC_DCIL_fetch import SAC
from tqc import *

from demo_extractor.demo_extractor_humanoid import DemoExtractor
from evaluate.humanoidenv.evaluate_humanoidenv import eval_trajectory_humanoid
from callbacks.callbacks import LogCallbackHumanoidEnv

from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
# from SAC_utils.vec_normalize_fetch import VecNormalize

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor



def learn_DCIL(args, env, eval_env, path):
    """
    Learn goal-conditioned policy for Goal Guided Imitation
    """

    if args["RL_algo"] == "SAC_HER":
        # Available strategies (cf paper): future, final, episode
        goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
        # If True the HER transitions will get sampled online
        online_sampling = True

        # Time limit for the episodes
        max_episode_length = args["max_episode_length"]
        ##### Warning: should it be fixed or can it be variable

        # Add action noise for "more i.i.d" transitions?
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = SAC("MultiInputPolicy", env,
                                        learning_rate = args["lr"],
                                        replay_buffer_class=HerReplayBuffer,
                                        # Parameters for HER
                                        replay_buffer_kwargs=dict(
                                        n_sampled_goal=4,
                                        goal_selection_strategy=goal_selection_strategy,
                                        online_sampling=online_sampling,
                                        max_episode_length=max_episode_length,
                                        ),
                                        #action_noise = action_noise,
                                        ent_coef=args["alpha_ent"],
                                        policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300], optimizer_kwargs={"eps":args["eps_optimizer"]}),#net_arch=[256, 256, 256]),
                                        #policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300], optimizer_class=torch.optim.RMSprop, optimizer_kwargs=dict(eps=args["eps_optimizer"])),
                                        verbose=1,
                                        warmup_duration=100,
                                        make_logs = True,
                                        path = path,
                                        bonus_reward_bool = args["bonus_reward_bool"],
                                        add_ent_reg_critic = args["add_ent_reg"],
                                        device= device)

        # set up logger for tensorboard (access tensorboard with cd /tmp/ && tensorboard --logdir sb3_log/)
        tmp_path = "/tmp/sb3_log/"
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

    if args["RL_algo"] == "TQC_HER":
        # Available strategies (cf paper): future, final, episode
        goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
        # If True the HER transitions will get sampled online
        online_sampling = True

        # Time limit for the episodes
        max_episode_length = args["max_episode_length"]
        ##### Warning: should it be fixed or can it be variable

        # Add action noise for "more i.i.d" transitions?
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = TQC("MultiInputPolicy", env,
                                        learning_rate = args["lr"],
                                        gamma = 0.95,
                                        batch_size = 1024,
                                        learning_starts = 1000,
                                        replay_buffer_class=HerReplayBuffer,
                                        # Parameters for HER
                                        replay_buffer_kwargs=dict(
                                        n_sampled_goal=4,
                                        goal_selection_strategy=goal_selection_strategy,
                                        online_sampling=online_sampling,
                                        max_episode_length=max_episode_length,
                                        ),
                                        ent_coef=args["alpha_ent"],
                                        policy_kwargs = dict(log_std_init=-3, net_arch=[512, 512, 512]),
                                        #policy_kwargs = dict(log_std_init=-3, net_arch=[400, 300], optimizer_class=torch.optim.RMSprop, optimizer_kwargs=dict(eps=args["eps_optimizer"])),
                                        verbose=1,
                                        device= device,
                                        add_bonus_reward = args["bonus_reward_bool"],
                                        add_ent_reg_critic = args["add_ent_reg"])

        # set up logger for tensorboard (access tensorboard with cd /tmp/ && tensorboard --logdir sb3_log/)
        tmp_path = "/tmp/sb3_log/"
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

    callback = LogCallbackHumanoidEnv(path, args["algo_type"], eval_env)

    total_timesteps, callback = model._setup_learn(args["total_timesteps"], eval_env, callback, -1, 5, args["RL_algo"], True, args["algo_type"])

    ## additional log files
    f_memory_usage = open(path + "/memory_usage.txt", "w")
    f_RAM_memory = open(path + "/RAM_memory.txt", "w")
    f_memory_percent = open(path + "/memory_percent.txt", "w")
    f_sr_skills = open(path + "/sr_skills.txt", "w")
    f_ratio = open(path + "/ratio.txt", "w")
    f_nb_skill_succeeded = open(path + "/nb_skill_succeeded.txt", "w")
    f_nb_skills_feasible = open(path + "/nb_skills_feasible.txt", "w")

    max_rollout_collection = 200
    last_episode_num = 0
    rollout_collection_cnt = 0
    timesteps_cnt = 0

    successful_traj = False

    callback.on_training_start(locals(), globals())

    ############################################################################
    ########################
    ########################            Training loop
    ########################
    ############################################################################

    while model.num_timesteps < total_timesteps:

        ## memory prints
        # process = psutil.Process(os.getpid())
        # print(" MEMORY USAGE : ", process.memory_info().rss / 1024 ** 2)
        # # print(" MEMORY USAGE : ", process.memory_info())
        # print('RAM memory % used:', psutil.virtual_memory()[2])
        # print(" MEMORY PERCENT : ", process.memory_percent())

        rollout_collection_cnt += 1
        timesteps_cnt += 1

        if args["algo_type"] == "OnPolicyAlgorithm":
            continue_training = model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=model.n_steps)

        if args["algo_type"] == "OffPolicyAlgorithm":
            # print("model.train_freq = ", model.train_freq
            # print("traj = ", callback.callbacks[0].trajs)
            rollout = model.collect_rollouts(model.env, train_freq = model.train_freq,
                                                        action_noise = model.action_noise,
                                                        callback = callback,
                                                        learning_starts = model.learning_starts,
                                                        replay_buffer = model.replay_buffer,
                                                        log_interval = 10)

            continue_training = rollout.continue_training

        if continue_training is False:
            break

        if rollout_collection_cnt > max_rollout_collection:
            print("nb of successfull rollouts = ", callback.callbacks[0].sum_W)
            print("total nb of rollouts = ", callback.callbacks[0].n_runs)
            sum_W = callback.callbacks[0].sum_W
            n_runs = callback.callbacks[0].n_runs

            print("skills feasibility = ", env.envs[0].skill_manager.L_skills_feasible)
            print("overshoot feasibility = ", env.envs[0].skill_manager.L_overshoot_feasible)

            nb_skills_feasible = sum([int(skill_feasible) for skill_feasible in env.envs[0].skill_manager.L_skills_feasible])
            f_nb_skills_feasible.write(str(nb_skills_feasible) + "\n")

            callback.callbacks[0].sum_W = 0
            callback.callbacks[0].n_runs = 0

            if float(n_runs) > 0.:
                ratio = float(sum_W) / float(n_runs)
            else:
                ratio = 0.

            f_ratio.write(str(ratio) + "\n")
            print("success ratio =  ", ratio)

            traj, skills_successes = eval_trajectory_humanoid(env, eval_env, model,
                                                                        args["algo_type"],
                                                                        callback.callbacks[0].path,
                                                                        callback.callbacks[0].nb_rollout,
                                                                        False,
                                                                        video = args["video"])

            print("full evaluation success = ", skills_successes)
            successfull_traj = skills_successes[-1]
            print("full evaluation success = ", successful_traj)


            #print("model._vec_normalize_env = ", model._vec_normalize_env.obs_rms["observation"].mean[:10])

            #with open(path+"/vec_normalize_env.pkl","wb") as f:
            #    pickle.dump(model._vec_normalize_env, f)


            f_nb_skill_succeeded.write(str(sum([int(skill_success) for skill_success in skills_successes])) + "\n")

            if successful_traj :
                traj, skills_successes = eval_trajectory_humanoid(env, eval_env,
                                                            model,
                                                            args["algo_type"],
                                                            callback.callbacks[0].path,
                                                            callback.callbacks[0].nb_rollout,
                                                            True,
                                                            video = args["video"])

                callback.callbacks[0]._visu_trajectories(eval_env,
                                                        eval_traj,
                                                        [],
                                                        callback.callbacks[0].trajs,
                                                        callback.callbacks[0].success_trajs)
                break

        if args["algo_type"] == "OnPolicyAlgorithm":
            model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
            model.train()

        if args["algo_type"] == "OffPolicyAlgorithm":
            if model.num_timesteps > 0 and model.num_timesteps > model.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = model.gradient_steps if model.gradient_steps > 0 else rollout.episode_timesteps
                model.train(batch_size=model.batch_size, gradient_steps=gradient_steps)

        if rollout_collection_cnt > max_rollout_collection:
            rollout_collection_cnt = 0

        if rollout_collection_cnt % 10 == 0:
            process = psutil.Process(os.getpid())
            f_memory_usage.write(str(process.memory_info().rss / 1024 ** 2) + "\n")
            f_RAM_memory.write(str(psutil.virtual_memory()[2]) + "\n")
            f_memory_percent.write(str(process.memory_percent()) + "\n")

    ## close log files
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
    model.f_critic_errors.close()
    model.f_critic_errors = None


    ## save model
    model.save(path + "/GCP_model")

    # f.close()
    # f_m_r.close()
    # f_m_l.close()
    f_ratio.close()
    f_memory_usage.close()
    f_RAM_memory.close()
    f_memory_percent.close()
    f_nb_skill_succeeded.close()
    f_nb_skills_feasible.close()

    total_nb_timesteps = model.num_timesteps
    callback.on_training_end()


    return total_nb_timesteps

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument for GCP imitation.')
    parser.add_argument('--eps_dist', help='espilon distance for cleaning trajectory')
    parser.add_argument('--algo', help='RL algorithm')
    parser.add_argument('--env', help='environment')
    parser.add_argument('--demo_path', help='path to demo file')
    parser.add_argument('--bonus_bool', help='add bonus reward')
    parser.add_argument('--overshoot_bool', help='overshoot if success yes(1) no(0)')
    parser.add_argument('--eps_optimizer', help='epsilon for adam optimizer')
    parser.add_argument('--alpha_ent', help='temperature coefficient for entropy regularization')
    parser.add_argument('-l', help='learning rate')
    parser.add_argument('-x', help='demo indx')
    parser.add_argument('--ent_reg_bool', help='add entropy regularization term for critic update')


    parsed_args = parser.parse_args()

    args = {}
    args["success_dist"] = 0.2
    args["success_ratio"] = 0.98
    args["num_episodes"] = 100
    args["episode_timesteps"] = 200
    args["normalization_timesteps"] = 200

    args["RL_algo"] = str(parsed_args.algo)
    args["env_name"] = str(parsed_args.env)
    args["demo_directory"] = str(parsed_args.demo_path)
    args["vec_norm_directory"] = str(parsed_args.demo_path) + "vec_norm_env_set/"
    args["demo_indx"] = int(parsed_args.x)

    args["demo_filename"] = args["demo_directory"] + str(args["demo_indx"]) + ".demo"
    # args["demo_filename"] = args["demo_directory"] + "/" + str(np.random.randint(1,7)) + ".demo"

    args["eps_dist"] = float(parsed_args.eps_dist)
    args["bonus_reward_bool"] = bool(int(parsed_args.bonus_bool))
    args["do_overshoot"] = bool(int(parsed_args.overshoot_bool))
    args["eps_optimizer"] = float(parsed_args.eps_optimizer)
    args["alpha_ent"] = float(parsed_args.alpha_ent)
    args["video"] = False
    args["total_timesteps"] = 400000 #600000
    args["lr"] = float(parsed_args.l)
    args["add_ent_reg"] = bool(int(parsed_args.ent_reg_bool))

    if "DDPG" in args["RL_algo"] or "SAC" in args["RL_algo"] or "TD3" in args["RL_algo"] or "TQC" in args["RL_algo"]:
        args["algo_type"] = "OffPolicyAlgorithm"
    else:
        args["algo_type"] = "OnPolicyAlgorithm"


    # create new directory
    now = datetime.now()
    dt_string = '_%s_%s' % (datetime.now().strftime('%Y%m%d'), str(os.getpid()))

    cur_path = os.getcwd()
    dir_path = cur_path + "/xp/DCIL_" + args["env_name"] + "_" + args["RL_algo"] + "_" + str(args["lr"]) + "_" + str(args["alpha_ent"]) + "_" + str(args["add_ent_reg"]) + "_" + str(args["demo_indx"]) + dt_string
    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s " % dir_path)

    ## extract environment option (goal space definition)
    env_name = args["env_name"].split("_")[0]
    if len(args["env_name"].split("_")) > 1:
        env_option = args["env_name"].split("_")[1]
    else:
        env_option = ""

    env_args = {"env_option":env_option , "do_overshoot":args["do_overshoot"]}

    print("env_option = ", env_option)

    if "HER" in args["RL_algo"] or "GCP" in args["RL_algo"]:
        # create local environment
        env_name = env_name + "GCPHERSB3-v0"
    else:
        env_name = env_name + "GCPSB3-v0"

    demo_extractor = DemoExtractor(dir_path, args["demo_filename"], env_name = env_name, env_args = env_args, eps_state = args["eps_dist"])
    demo_extractor.visu_demonstration()

    ## create environment
    env = demo_extractor.get_env()

    assert len(env.skill_manager.L_inner_states) == len(env.skill_manager.L_states)
    assert len(env.skill_manager.starting_state_set) == len(env.skill_manager.L_states)
    args["max_episode_length"] = max(env.skill_manager.L_budgets)
    # print("max_episode_length = ", args["max_episode_length"])
    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, env_args["env_option"], norm_obs=True, norm_reward=False, clip_obs=np.inf)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=np.inf)

    ## load default vec normalize env
    # with open(args["vec_norm_directory"]+"/vec_normalize_env_" + str(args["demo_indx"]) + "_random.pkl","rb") as f:
    #     default_vec_normalize_env = pickle.load(f)
    #
    # ## fix mean and var according to default vec normalize env + disable training i.e mean and var are fixed
    # env.obs_rms["observation"].mean = default_vec_normalize_env.obs_rms["observation"].mean
    # env.obs_rms["observation"].var = default_vec_normalize_env.obs_rms["observation"].var
    # env.obs_rms["achieved_goal"].mean = default_vec_normalize_env.obs_rms["achieved_goal"].mean
    # env.obs_rms["achieved_goal"].var = default_vec_normalize_env.obs_rms["achieved_goal"].var
    # env.obs_rms["desired_goal"].mean = default_vec_normalize_env.obs_rms["desired_goal"].mean
    # env.obs_rms["desired_goal"].var = default_vec_normalize_env.obs_rms["desired_goal"].var
    # env.training = False

    env = VecMonitor(env)
    # env = VecNormalize(env, norm_obs=True, norm_reward=False)



    ## create eval environment
    eval_env = demo_extractor.get_env()
    assert len(eval_env.skill_manager.L_inner_states) == len(eval_env.skill_manager.L_states)
    assert len(eval_env.skill_manager.starting_state_set) == len(eval_env.skill_manager.L_states)

    ## reinitialize random seed
    random.seed(None)

    # start learning
    total_nb_timesteps = learn_DCIL(args, env, eval_env, dir_path)

    f = open(dir_path + "/nb_timesteps.txt", "a")
    f.write(str(total_nb_timesteps))
    f.close()
