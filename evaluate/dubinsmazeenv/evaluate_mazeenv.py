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

from datetime import datetime

import gym
import numpy as np

import torch


def eval_trajectory_mazeenv(env, eval_env, model, algo_type):
    """
    Evaluate agent on the full trajectory (chaining of all skills)
    Iterate over skills and wait for success of time step limit to shift to the next skill.
    Return successful eval boolean iff the last skill has been achieved.
    """

    ## reset to first skill
    eval_env.reset()
    eval_env.reset_skill_by_nb(1)

    traj = [eval_env.get_state()]

    eval_env.testing = True

    skills_successes = []
    max_zone = 1

    while eval_env.skill_manager.indx_goal <= eval_env.skill_manager.nb_skills:

        skill_success = False

        for i_step in range(0,eval_env.max_steps):

            ## Warning: depends on the algorithm
            obs = eval_env._get_obs()
            new_zone = eval_zone(obs["observation"])
            if new_zone > max_zone:
                max_zone = new_zone

            ## normalize goal according to training environment
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
                skill_success = True
                break

        ## change goal and max_steps
        eval_env.next_skill()

        skills_successes.append(skill_success)

    return traj, skills_successes, max_zone

def eval_zone(state):
    x = state[0]
    y = state[1]
    if y < 1.:
        if x < 1.:
            return 1
        elif  x < 2.:
            return 2
        elif  x < 3.:
            return 3
        elif  x < 4.:
            return 4
        else:
            return 5
    elif y < 2.:
        if  x > 4.:
            return 6
        elif  x > 3.:
            return 7
        elif x > 2.:
            return 8
        else:
            return 11
    elif y < 3.:
        if x < 1.:
            return 11
        elif x < 2.:
            return 10
        elif x < 3.:
            return 9
        elif x < 4.:
            return 20
        else :
            return 21

    elif y < 4.:
        if x < 1.:
            return 12
        elif x < 2.:
            return 15
        elif x < 3.:
            return 16
        elif x < 4:
            return 19
        else :
            return 22
    else:
        if x < 1.:
            return 13
        elif x < 2.:
            return 14
        elif x < 3.:
            return 17
        elif x < 4:
            return 18
        else :
            return 23
