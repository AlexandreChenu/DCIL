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

def save_frames_as_video(frames, path, iteration):

    video_name = path + '/video' + str(iteration) + '.mp4'
    height, width, layers = frames[0].shape
    #resize
    percent = 30
    width = int(frames[0].shape[1] * percent / 100)
    height = int(frames[0].shape[0] * percent / 100)

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

    for frame in frames:
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        resize_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        cvt_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)
        video.write(cvt_frame)

    video.release()
    cv2.destroyAllWindows()

    return

def eval_trajectory_humanoidenv(env, eval_env, model, algo_type, path, nb_rollout, testing, video=False):
    """
    Evaluate agent on the full trajectory (chaining of all tasks)
    Iterate over tasks and wait for success of time step limit to shift to the next task.
    Return successful eval boolean iff the last task has been achieved.
    """

    eval_env.reset()
    eval_env.reset_task_by_nb(1)

    traj = [eval_env.env.get_state()]

    eval_env.testing = True

    tasks_successes = []

    frames = []

    while eval_env.tasks.indx_goal < eval_env.tasks.nb_tasks + 1:

        task_success = False

        for i_step in range(0,eval_env.max_steps):
            if video:
                frame = eval_env.env.env.sim.render(mode="offscreen", width=1980, height= 1080)
                frames.append(frame)

            ## Warning: depends on the algorithm
            obs = eval_env._get_obs()

            ## normalize obs
            obs = env.normalize_obs(obs)

            obs["observation"] = torch.FloatTensor([obs["observation"]])
            obs["desired_goal"] = torch.FloatTensor([obs["desired_goal"]])
            obs["achieved_goal"] = torch.FloatTensor([obs["achieved_goal"]])

            if algo_type == "OffPolicyAlgorithm":
                action, _ = model.predict(obs, deterministic=True)
                action = action[0]

            else:
                action, _, _ = model.forward(obs, deterministic = True).detach().numpy()[0]

            new_obs, _, done, info = eval_env.step_test(action)
            traj.append(new_obs["observation"])

            if info['target_reached']:
                task_success = True
                break

        ## change goal and max_steps
        eval_env.advance_task()

        tasks_successes.append(task_success)

        if nb_rollout % 10000 == 0 or testing:
            if video:
                save_frames_as_video(frames, path, nb_rollout)

    return traj, tasks_successes[-1]

def eval_tasks_humanoidenv(env, eval_env, model, algo_type):

    eval_env.reset()

    L_traj = []

    eval_env.testing = True

    for i in range(1,eval_env.tasks.nb_tasks+1):

        eval_env.reset_task_by_nb(i)

        traj = [copy.deepcopy(eval_env.env.get_state())]

        for i_step in range(0,eval_env.max_steps):

            ## Warning: depends on the algorithm
            obs = eval_env._get_obs()

            ## normalize obs
            obs = env.normalize_obs(obs)

            obs["observation"] = torch.FloatTensor([obs["observation"]])
            obs["desired_goal"] = torch.FloatTensor([obs["desired_goal"]])
            obs["achieved_goal"] = torch.FloatTensor([obs["achieved_goal"]])


            if algo_type == "OffPolicyAlgorithm":
                action, _ = model.predict(obs, deterministic=True)
                #print("action = ", action)
                # unscaled_action = model.policy.actor.predict(obs, deterministic=True).detach().numpy()[0]
                #action = model.policy.unscale_action(unscaled_action)
                action = action[0]
            else:
                action, _, _ = model.policy.forward(obs, deterministic = True).detach().numpy()[0]

            new_obs, _, done, info = eval_env.step_test(action)
            traj.append(new_obs["observation"])

            if info['target_reached']:
                break

        L_traj.append(traj)
        traj = []

    return L_traj
