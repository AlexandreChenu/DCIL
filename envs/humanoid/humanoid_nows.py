# Modifications Copyright (c) 2020 Uber Technologies Inc.
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
import mujoco_py

"""
Modified from https://github.com/openai/mlsh from
the "Meta-Learning Shared Hierarchies" paper: https://arxiv.org/abs/1710.09767
"""

class HumanoidEnvNOWS(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        this_path = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, this_path + '/assets/humanoidstandup_nows.xml', 5)
        utils.EzPickle.__init__(self)

        # self.qpos_start = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        # self.qvel_start = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        self.goal = np.array([1., 1.])

        # self._set_action_space()
        #
        # action = self.action_space.sample()
        # observation, _, done, _ = self.step(action)
        #
        # print("observation = ", observation)
        # assert not done
        #
        # self._set_observation_space(observation['observation'])

    def reward_function(self, obs, goal):
        d =  np.linalg.norm(obs - goal, axis=-1)
        return -d

    # def step(self, a):
    #
    #     # get pos before
    #     ob_b = self._get_obs()
    #     posbefore = ob_b[0]
    #
    #     self.do_simulation(a, self.frame_skip)
    #
    #     # get pos after
    #     ob_a = self._get_obs()
    #     posafter = ob_a[0]
    #
    #     reward = self.reward_function(self.data.qpos[:2], self.goal)
    #
    #     done = False
    #     ob = self._get_obs()
    #     return ob, reward, done, dict(bc=self.data.qpos[:2],
    #                                   x_pos=self.data.qpos[0])

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        pos_after = self.sim.data.qpos[2]
        data = self.sim.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        # reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        reward = 0

        done = bool(False)
        return self._get_obs(), reward, done, dict(reward_linup=uph_cost, reward_quadctrl=-quad_ctrl_cost, reward_impact=-quad_impact_cost)

    def _get_obs(self):
        data = self.sim.data

        print(data.qpos.flat[:3])
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # def reset_model(self):
    #     qpos = self.qpos_start
    #     qvel = self.qvel_start
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()

    # get simulation state
    def get_state_sim(self):
        return self.sim.get_state()

    # set simulation state (more than just the position and velocity) to saved one
    def set_state_sim(self, saved_state):
        self.sim.set_state(saved_state)
        #return self.sim.get_state()
        return self._get_obs()


    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20
