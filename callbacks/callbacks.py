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
import seaborn
seaborn.set()
seaborn.set_style("whitegrid")

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import collections as mc
import matplotlib as mpl
import matplotlib.animation as animation

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

from stable_baselines3.common.callbacks import BaseCallback
from evaluate.dubinsmazeenv.evaluate_mazeenv import *
from evaluate.fetchenv.evaluate_fetchenv import *
from evaluate.humanoidenv.evaluate_humanoidenv import *

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")



class LogCallbackMazeEnv(BaseCallback):
    """
    Callback for saving visual log in Maze environments
    """

    def __init__(self, path, algo, algo_type, env, verbose: int = 0):
        super().__init__(verbose)
        self.trajs = []
        self.traj = []
        self.success_trajs = []
        self.eval_trajs = []
        self.nb_rollout = 0
        self.sum_W = 0
        self.n_runs = 0
        self.path = path
        self.algo = algo
        self.algo_type = algo_type

        if self.algo_type == "OffPolicyAlgorithm":
            # self.freq_rollout_display = 1000
            self.freq_rollout_display = 1000 / 3
            # self.freq_rollout_display = 100
            self.freq_eval_adapted_traj = 50
        else:
            self.freq_rollout_display = 10

    def _on_step(self) -> bool:
        # print("self.locals.keys() = ", self.locals.keys())

        info = self.locals["infos"][0]
        eval_env = self.locals["eval_env"]

        if info['done']:
            if info['target_reached']:
                self.sum_W += 1
                self.success_trajs.append(info['traj'])
            self.n_runs += 1
            self.trajs.append(info['traj'])
        return True


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        After running a collect_rollouts for nb_steps timesteps.
        """

        self.nb_rollout += 1
        eval_env = self.locals["eval_env"]
        env = self.locals["env"]

        if self.nb_rollout % self.freq_rollout_display == 0:

            eval_traj, successful_traj, max_zone = eval_trajectory_mazeenv(env, eval_env, self.model, self.algo_type)
            self.eval_trajs.append(eval_traj)
            self._visu_trajectories(env, eval_env, [eval_traj], self.trajs, [])
            self._visu_V()
            self.eval_trajs = []
            self.trajs = []
            self.success_trajs = []

        return True


    def _visu_trajectories(self, env, eval_env, eval_trajs, training_trajs, success_trajs):
        """
        Visualization of the demonstration, rollout trajectories
        and the complete evaluation of the agent.
        """
        fig = plt.figure()
        fig.set_figheight(5.2)
        fig.set_figwidth(5.2)
        ax = fig.add_subplot()

        ## draw environment
        eval_env.draw(ax, paths=False)

        ## plot subgoals extracted from the demonstration
        L_X = [state[0] for state in eval_env.skill_manager.L_states]
        L_Y = [state[1] for state in eval_env.skill_manager.L_states]
        L_theta = [state[2] for state in eval_env.skill_manager.L_states]


        ## plot unsuccessful training rollouts
        for indx, traj in enumerate(training_trajs):
            if indx % 2 == 0:
                age = indx
                alpha = float(age/len(training_trajs))*0.8
                X_traj = [state[0] for state in traj]
                Y_traj = [state[1] for state in traj]
                ax.plot(X_traj, Y_traj, color = "lightsteelblue", alpha = alpha ,zorder = 1)

        circles = []
        for i in range(1,len(L_X)-1):
            circle = plt.Circle((L_X[i], L_Y[i]), eval_env.width_success, color='m', alpha = 0.6)
            circles.append(circle)
            # ax.add_patch(circle)

        circle = plt.Circle((L_X[len(L_X)-1], L_Y[len(L_X)-1]), eval_env.width_success, color='g', alpha = 0.6)
        circles.append(circle)
        # ax.add_patch(circle)
        coll = mc.PatchCollection(circles, color="plum", zorder = 4)
        ax.add_collection(coll)

        for indx, eval_traj in enumerate(eval_trajs):
            X_eval = [eval_env.project_to_goal_space(state)[0] for state in eval_traj]
            Y_eval = [eval_env.project_to_goal_space(state)[1] for state in eval_traj]
            for state in eval_traj:
                eval_env.plot_car(state, ax, alpha = 0.7, cabcolor="royalblue", truckcolor="royalblue")
            ax.plot(X_eval, Y_eval, color = "royalblue", alpha = 0.7, zorder = 3)

        for i in range(1,len(L_X)-1):
            x = L_X[i]
            y = L_Y[i]
            dx = np.cos(L_theta[i])
            dy = np.sin(L_theta[i])
            arrow = plt.arrow(x,y,dx*0.3,dy*0.3,alpha = 0.6,width = 0.04, color="m", zorder=6)
            ax.add_patch(arrow)

        plt.savefig(self.path + "/iteration_" + str(self.nb_rollout) + ".png")
        #plt.show()
        plt.close(fig)

        return 0

    def _visu_V(self) -> None:
        ## retrieve env
        eval_env = self.locals["eval_env"]
        env = self.locals["env"]

        ## Toy DubinsMaze
        states = [eval_env.skill_manager.L_states[0], eval_env.skill_manager.L_states[1]]
        goal_indxs = [1,2]
        desired_goals = [eval_env.skill_manager.L_states[1], eval_env.skill_manager.L_states[2]]
        orientation_ranges = [[-np.pi/2,np.pi/2], [0.,np.pi]]

        for state, goal_indx, desired_goal, orientation_range in zip(states, goal_indxs, desired_goals, orientation_ranges):
            self._visu_value_function_cst_speed(env, eval_env, state, goal_indx, desired_goal, orientation_range)
        return

    def _visu_value_function_cst_speed(self, env, eval_env, state, goal_indx, desired_goal, orientation_range):

        # s_v = np.linspace(0.5,1.,100)
        s_theta = np.linspace(orientation_range[0],orientation_range[1],100)
        values = []

        for theta in list(s_theta):
            obs = eval_env.get_obs()
            #print("action = ", action)
            obs["observation"][0] = state[0]
            obs["observation"][1] = state[1]
            obs["observation"][2] = theta

            obs["achieved_goal"][0] = state[0]
            obs["achieved_goal"][1] = state[1]

            obs["desired_goal"] = np.array([desired_goal[0],desired_goal[1]])

            ### Normalization
            obs["observation"] = np.array([obs["observation"]])
            obs["desired_goal"] = np.array([obs["desired_goal"]])
            obs["achieved_goal"] = np.array([obs["achieved_goal"]])
            obs = env.normalize_obs(obs)
            ####

            obs["observation"] = torch.FloatTensor(obs["observation"]).to(device)
            obs["desired_goal"] = torch.FloatTensor(obs["desired_goal"]).to(device)
            obs["achieved_goal"] = torch.FloatTensor(obs["achieved_goal"]).to(device)

            action = self.model.actor._predict(obs, deterministic=True)

            # q_values = self.model.critic(obs, action)

            # # Compute the next values from quantiles
            if "SAC" in self.algo:
                value = self.model.critic(obs, action)[0]
            elif "TQC" in self.algo:
                value = self.model.critic_target(obs, action).mean(dim=2).mean(dim=1, keepdim=True)
            else:
                value = None
                print("Unknown RL algo")

            # print("q_values[0] = ", q_values[0])
            values.append(value.detach().cpu().numpy()[0])

        fig, ax = plt.subplots()
        plt.plot(list(s_theta), values,label="learned V(s,g')")
        plt.plot()
        plt.xlabel("theta")
        plt.ylabel("value")
        plt.legend()

        plt.savefig(self.path + "/visu_values_" + str(goal_indx) + "_it_" + str(self.nb_rollout) + ".png")
        #plt.show()
        plt.close(fig)

        return

class LogCallbackFetchEnv(BaseCallback):
    """
    Callback for saving visual log in Fetch environments
    """

    def __init__(self, path, algo_type, env, verbose: int = 0):
        super().__init__(verbose)
        self.trajs = []
        self.traj = []
        self.success_trajs = []
        self.nb_steps = 0
        self.total_nb_steps = 0
        self.sum_W = 0
        self.n_runs = 0
        self.path = path
        self.algo_type = algo_type

        self.values_logfile_agent = open(self.path + "/values_logs_agent.txt", "w")

        if self.algo_type == "OffPolicyAlgorithm":
            self.freq_rollout_display = 1000
            # self.freq_rollout_display = 100
            self.freq_eval_adapted_traj = 50

        else:
            self.freq_rollout_display = 10

        ## save videos?
        self.video = False


    def _on_step(self) -> bool:
        # print("self.locals.keys() = ", self.locals.keys())

        info = self.locals["infos"][0]
        eval_env = self.locals["eval_env"]
        env = self.locals["env"]

        self.nb_steps += 1
        self.total_nb_steps += 1

        if info['done']:
            if info['target_reached']:
                self.sum_W += 1
                self.success_trajs.append(info['traj'])
            self.n_runs += 1
            self.trajs.append(info['traj'])

        return True


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        After running a collect_rollouts for nb_steps timesteps.
        """

        ## extract eval env
        eval_env = self.locals["eval_env"]
        env = self.locals["env"]

        if self.nb_steps > self.freq_rollout_display :

            ## eval full trajectory
            eval_traj,_,_ = eval_trajectory_fetchenv(env, eval_env, self.model, self.algo_type, self.path, self.total_nb_steps, False, video=self.video)
            ## eval each individual skill
            eval_skills = eval_skills_fetchenv(env, eval_env, self.model, self.algo_type)

            self._visu_trajectories(eval_env, eval_traj, eval_skills, self.trajs, self.success_trajs)
            # self._visu_value_function(env, eval_env)

            self.trajs = []
            self.success_trajs = []

            self.nb_steps = 0

        return True


    def _visu_success_zones(self, eval_env, ax):
        """
        Visualize success zones as sphere of radius eps_success around skill-goals
        """
        L_states = copy.deepcopy(eval_env.skill_manager.L_states)

        for state in L_states:
            goal = eval_env.project_to_goal_space(state)

            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

            x = goal[0] + 0.075*np.cos(u)*np.sin(v)
            y = goal[1] + 0.075*np.sin(u)*np.sin(v)
            z = goal[2] + 0.075*np.cos(v)
            ax.plot_wireframe(x, y, z, color="blue", alpha = 0.1)

        return


    def _visu_trajectories(self, eval_env, eval_traj, eval_skill, training_trajs, success_trajs):
        """
        Plot demonstration, full evaluation trajectory, training trajectory.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ## plot training traj
        for trajs in training_trajs:
            traj = trajs[0]
            ## gripper traj
            X_traj = [state[0] for state in traj]
            Y_traj = [state[1] for state in traj]
            Z_traj = [state[2] for state in traj]
            ax.plot(X_traj, Y_traj, Z_traj, color = "lightsteelblue", alpha = 0.35)

            traj = trajs[1]
            ## object traj
            X_traj = [state[0] for state in traj]
            Y_traj = [state[1] for state in traj]
            Z_traj = [state[2] for state in traj]
            ax.plot(X_traj, Y_traj, Z_traj, color = "steelblue", alpha = 0.35)

        for trajs in success_trajs:
            traj = trajs[0]
            X_traj = [state[0] for state in traj]
            Y_traj = [state[1] for state in traj]
            Z_traj = [state[2] for state in traj]
            ax.plot(X_traj, Y_traj, Z_traj, color = "red", alpha = 0.35)

        # ## scatter plot current goal
        # ax.scatter(eval_env.goal[0], eval_env.goal[1], eval_env.goal[2], color = "red", alpha = 0.5 )

        ## scatter plot demo
        demo = eval_env.skill_manager.L_states

        # for state in demo: ## differentiate states w/ grasping and states w/o
        #
        #     grasping_bool = eval_env.check_grasping(state)
        #     if grasping_bool:
        #         ax.scatter(eval_env.project_to_goal_space(state)[0], eval_env.project_to_goal_space(state)[1], eval_env.project_to_goal_space(state)[2], color = "saddlebrown", alpha = 0.8)
        #     else:
        #         ax.scatter(eval_env.project_to_goal_space(state)[0], eval_env.project_to_goal_space(state)[1], eval_env.project_to_goal_space(state)[2], color = "blue", alpha = 0.8)

        self._visu_success_zones(eval_env, ax)

        # if "full" in eval_env.env_option:
        X_demo_object = [eval_env.get_object_pos(state)[0] for state in demo]
        Y_demo_object = [eval_env.get_object_pos(state)[1] for state in demo]
        Z_demo_object = [eval_env.get_object_pos(state)[2] for state in demo]
        ax.scatter(X_demo_object, Y_demo_object, Z_demo_object, color = "gray", alpha = 0.8)

        full_demo = eval_env.skill_manager.L_full_demonstration
        # plot gripper traj in full demo
        X_demo = [eval_env.project_to_goal_space(state)[0] for state in full_demo]
        Y_demo = [eval_env.project_to_goal_space(state)[1] for state in full_demo]
        Z_demo = [eval_env.project_to_goal_space(state)[2] for state in full_demo]
        ax.plot(X_demo, Y_demo, Z_demo, color = "blue", alpha = 0.8)

        # plot object traj in full demo
        X_demo_object = [eval_env.get_object_pos(state)[0] for state in full_demo]
        Y_demo_object = [eval_env.get_object_pos(state)[1] for state in full_demo]
        Z_demo_object = [eval_env.get_object_pos(state)[2] for state in full_demo]
        ax.plot(X_demo_object, Y_demo_object, Z_demo_object, color = "gray", alpha = 0.8)


        ## plot skills
        for skill in eval_skill:
            X_skill_gripper = [eval_env.get_gripper_pos(state)[0] for state in skill]
            Y_skill_gripper = [eval_env.get_gripper_pos(state)[1] for state in skill]
            Z_skill_gripper = [eval_env.get_gripper_pos(state)[2] for state in skill]
            ax.plot(X_skill_gripper, Y_skill_gripper, Z_skill_gripper, color = "black", alpha = 0.8)

            # if "full" in eval_env.env_option:
            X_skill_object = [eval_env.get_object_pos(state)[0] for state in skill]
            Y_skill_object = [eval_env.get_object_pos(state)[1] for state in skill]
            Z_skill_object = [eval_env.get_object_pos(state)[2] for state in skill]
            ax.plot(X_skill_object, Y_skill_object, Z_skill_object, color = "orange", alpha = 0.8)

        ## plot gripper traj
        X_eval = [eval_env.get_gripper_pos(state)[0] for state in eval_traj]
        Y_eval = [eval_env.get_gripper_pos(state)[1] for state in eval_traj]
        Z_eval = [eval_env.get_gripper_pos(state)[2] for state in eval_traj]
        ax.plot(X_eval, Y_eval, Z_eval, color = "red", alpha = 1.)

        # plot object traj
        X_eval_object = [eval_env.get_object_pos(state)[0] for state in eval_traj]
        Y_eval_object = [eval_env.get_object_pos(state)[1] for state in eval_traj]
        Z_eval_object = [eval_env.get_object_pos(state)[2] for state in eval_traj]
        ax.plot(X_eval_object, Y_eval_object, Z_eval_object, color = "green", alpha = 1.)

        ax.set_xlim((0., 1.5))
        ax.set_ylim((-1., 2.))
        ax.set_zlim((0., 2.2))

        #ax.set_xlim((0.5, 1.2))
        #ax.set_ylim((-0.5, 1.5))
        #ax.set_zlim((0., 1.5))

        ## plot 4 different orientation
        for azim_ in range(45,360,90):
            ax.view_init(azim = azim_)
            plt.savefig(self.path + "/iteration_" + str(azim_) + "_" + str(self.total_nb_steps)+ ".png")
            #for elev_ in [0,90]:
            #    ax.view_init(azim = azim_, elev=elev_)
            #    plt.savefig(self.path + "/iteration_" + str(azim_) + "_" + str(elev_) + "_" + str(self.nb_rollout)+ ".png")
        plt.close(fig)

        return 0

    def _update_trajectories(self, num, pos, lines):
        for line, p in zip(lines, p):
            print("pos = ", p[:num])
            line.set_data(p[:num,:2])
            line.set_3d_properties(p[:num, 2])
        return lines

    def _make_animation(self, eval_env, eval_traj, eval_skill):
        """
        Plot demonstration, full evaluation trajectory, training trajectory.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ## x,y,z lim
        ax.set_xlim((0., 1.5))
        ax.set_ylim((-1., 2.))
        ax.set_zlim((0., 2.2))

        ## plot 4 different orientation
        ax.view_init(azim = 315)

        # ## scatter plot current goal
        # ax.scatter(eval_env.goal[0], eval_env.goal[1], eval_env.goal[2], color = "red", alpha = 0.5 )

        # ## scatter plot demo
        # demo = eval_env.skills.L_states
        #
        # for state in demo: ## differentiate states w/ grasping and states w/o
        #     grasping_bool = eval_env.check_grasping(state)
        #     if grasping_bool:
        #         ax.scatter(eval_env.project_to_goal_space(state)[0], eval_env.project_to_goal_space(state)[1], eval_env.project_to_goal_space(state)[2], color = "saddlebrown", alpha = 0.8)
        #     else:
        #         ax.scatter(eval_env.project_to_goal_space(state)[0], eval_env.project_to_goal_space(state)[1], eval_env.project_to_goal_space(state)[2], color = "blue", alpha = 0.8)

        self._visu_success_zones(eval_env, ax)

        full_demo = eval_env.skill_manager.L_full_demonstration
        # plot gripper traj in full demo
        X_demo = [eval_env.project_to_goal_space(state)[0] for state in full_demo]
        Y_demo = [eval_env.project_to_goal_space(state)[1] for state in full_demo]
        Z_demo = [eval_env.project_to_goal_space(state)[2] for state in full_demo]
        ax.plot(X_demo, Y_demo, Z_demo, color = "gray", alpha = 0.8)

        # plot object traj in full demo
        X_demo_object = [eval_env.get_object_pos(state)[0] for state in full_demo]
        Y_demo_object = [eval_env.get_object_pos(state)[1] for state in full_demo]
        Z_demo_object = [eval_env.get_object_pos(state)[2] for state in full_demo]
        ax.plot(X_demo_object, Y_demo_object, Z_demo_object, color = "dimgray", alpha = 0.8)

        # ## plot gripper traj
        # X_eval = [eval_env.get_gripper_pos(state)[0] for state in eval_traj]
        # Y_eval = [eval_env.get_gripper_pos(state)[1] for state in eval_traj]
        # Z_eval = [eval_env.get_gripper_pos(state)[2] for state in eval_traj]
        # ax.plot(X_eval, Y_eval, Z_eval, color = "red", alpha = 1.)
        #
        # # plot object traj
        # X_eval_object = [eval_env.get_object_pos(state)[0] for state in eval_traj]
        # Y_eval_object = [eval_env.get_object_pos(state)[1] for state in eval_traj]
        # Z_eval_object = [eval_env.get_object_pos(state)[2] for state in eval_traj]
        # ax.plot(X_eval_object, Y_eval_object, Z_eval_object, color = "green", alpha = 1.)

        pos = [np.array([eval_env.get_gripper_pos(state)[0] for state in eval_traj]),
                np.array([eval_env.get_object_pos(state)[0] for state in eval_traj])]

        ## line 1: gripper traj, line 2: object pos
        lines = [ax.plot([], [], [])[0], ax.plot([], [], [])[0]]

        # Creating the Animation object
        ani = animation.FuncAnimation(
            fig, update_lines, len(eval_traj), fargs=(pos, lines), interval=200)

        ani.save(self.path + "/eval_anim_" + str(self.nb_rollout) + ".mp4")

        return 0

    def _visu_value_function(self, env, eval_env):

        L_states = eval_env.skill_manager.L_states

        values_agent = []
        values_expert = []

        # for state, action in zip(L_states, L_actions):
        for i in range(len(L_states)-1):
            obs = eval_env.get_obs()
            #print("action = ", action)
            obs["observation"][:] = L_states[i][:]
            obs["achieved_goal"] = eval_env.project_to_goal_space(L_states[i])
            obs["desired_goal"] = eval_env.project_to_goal_space(L_states[i+1])

            ### Normalization
            obs["observation"] = np.array([obs["observation"]])
            obs["desired_goal"] = np.array([obs["desired_goal"]])
            obs["achieved_goal"] = np.array([obs["achieved_goal"]])
            obs = env.normalize_obs(obs)
            ####

            obs["observation"] = torch.FloatTensor(obs["observation"]).to(device)
            obs["desired_goal"] = torch.FloatTensor(obs["desired_goal"]).to(device)
            obs["achieved_goal"] = torch.FloatTensor(obs["achieved_goal"]).to(device)

            action_agent = self.model.actor._predict(obs, deterministic=True)
            q_values_agent = self.model.critic(obs, action_agent)

            values_agent.append(q_values_agent[0].detach().cpu().numpy()[0])

        for value_agent, value_expert in zip(values_agent, values_expert):
            self.values_logfile_agent.write(str(value_agent)+ " ")

        self.values_logfile_agent.write("\n")
        self.values_logfile_agent.flush()

        return


class LogCallbackHumanoidEnv(BaseCallback):
    """
    Callback for saving visual log in Humanoid environments
    """

    def __init__(self, path, algo_type, env, verbose: int = 0):
        super().__init__(verbose)
        self.trajs = []
        self.traj = []
        self.success_trajs = []
        self.nb_rollout = 0
        self.sum_W = 0
        self.n_runs = 0
        self.path = path
        self.algo_type = algo_type

        self.values_logfile_agent = open(self.path + "/values_logs_agent.txt", "w")

        if self.algo_type == "OffPolicyAlgorithm":
            self.freq_rollout_display = 1000
            # self.freq_rollout_display = 100
            self.freq_eval_adapted_traj = 50

        else:
            self.freq_rollout_display = 10

        ## save videos?
        self.video = False


    def _on_step(self) -> bool:
        # print("self.locals.keys() = ", self.locals.keys())

        info = self.locals["infos"][0]
        eval_env = self.locals["eval_env"]
        env = self.locals["env"]

        if info['done']:
            if info['target_reached']:
                self.sum_W += 1
                self.success_trajs.append(info['traj'])
            self.n_runs += 1
            self.trajs.append(info['traj'])

        return True


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        After running a collect_rollouts for nb_steps timesteps.
        """

        self.nb_rollout += 1

        ## extract eval env
        eval_env = self.locals["eval_env"]
        env = self.locals["env"]

        if self.nb_rollout % self.freq_rollout_display == 0:

            ## eval full trajectory
            eval_traj,_ = eval_trajectory_humanoid(env, eval_env, self.model, self.algo_type, self.path, self.nb_rollout, False, video=self.video)
            ## eval each individual skill
            eval_skills = eval_skills_humanoidenv(env, eval_env, self.model, self.algo_type)

            self._visu_trajectories(eval_env, eval_traj, eval_skills, self.trajs, self.success_trajs)
            # self._visu_value_function(env, eval_env)

            self.trajs = []
            self.success_trajs = []

        return True


    def _visu_success_zones(self, eval_env, ax):
        """
        Visualize success zones as sphere of radius eps_success around skill-goals
        """
        L_states = copy.deepcopy(eval_env.skill_manager.L_states)

        for state in L_states:
            goal = eval_env.project_to_goal_space(state)

            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

            x = goal[0] + 0.075*np.cos(u)*np.sin(v)
            y = goal[1] + 0.075*np.sin(u)*np.sin(v)
            z = goal[2] + 0.075*np.cos(v)
            ax.plot_wireframe(x, y, z, color="blue", alpha = 0.1)

        return


    def _visu_trajectories(self, eval_env, eval_traj, eval_skills, training_trajs, success_trajs):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ## plot training traj
        for traj in training_trajs:
            X_traj = [state[0] for state in traj]
            Y_traj = [state[1] for state in traj]
            Z_traj = [state[2] for state in traj]
            ax.plot(X_traj, Y_traj, Z_traj, color = "pink", alpha = 0.35)

        for traj in success_trajs:
            X_traj = [state[0] for state in traj]
            Y_traj = [state[1] for state in traj]
            Z_traj = [state[2] for state in traj]
            ax.plot(X_traj, Y_traj, Z_traj, color = "red", alpha = 0.35)

        ## scatter plot current goal
        ax.scatter(eval_env.goal[0], eval_env.goal[1], eval_env.goal[2], color = "red", alpha = 0.5 )

        ## scatter plot demo
        demo = eval_env.skill_manager.L_states
        # grasping_bools = [eval_env.check_grasping(state) for state in demo]
        for state in demo: ## differentiate states w/ grasping and states w/o
            ax.scatter(eval_env.project_to_goal_space(state)[0], eval_env.project_to_goal_space(state)[1], eval_env.project_to_goal_space(state)[2], color = "blue", alpha = 0.8)

        self._visu_success_zones(eval_env, ax)

        full_demo = eval_env.skill_manager.L_full_demonstration
        X_demo = [eval_env.project_to_goal_space(state)[0] for state in full_demo]
        Y_demo = [eval_env.project_to_goal_space(state)[1] for state in full_demo]
        Z_demo = [eval_env.project_to_goal_space(state)[2] for state in full_demo]
        ax.plot(X_demo, Y_demo, Z_demo, color = "blue", alpha = 0.8)

        ## plot skills
        for skill in eval_skills:
            X_skill = [state[0] for state in skill]
            Y_skill = [state[1] for state in skill]
            Z_skill = [state[2] for state in skill]
            ax.plot(X_skill, Y_skill, Z_skill, color = "black", alpha = 0.8)

        ## scatter plot traj
        X_eval = [state[0] for state in eval_traj]
        Y_eval = [state[1] for state in eval_traj]
        Z_eval = [state[2] for state in eval_traj]
        ax.plot(X_eval, Y_eval, Z_eval, color = "red", alpha = 1.)

        # ax.set_xlim((0., 1.))
        ax.set_xlim((0., 5.))
        # ax.set_ylim((0., 2.))
        ax.set_ylim((0., -2))
        # ax.set_zlim((0, 1.))
        ax.set_zlim((0., 1.5))

        for azim_ in range(45,360,90):
            ax.view_init(azim = azim_)
            plt.savefig(self.path + "/iteration_" + str(azim_) + "_" + str(self.nb_rollout) + ".png")
        #plt.show()
        plt.close(fig)

        return 0
