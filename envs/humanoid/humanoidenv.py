import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        this_path = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, this_path + '/assets/humanoid.xml', 5)
        # mujoco_env.MujocoEnv.__init__(self, "humanoid.xml", 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat,
                data.qvel.flat,
                data.cinert.flat,
                data.cvel.flat,
                data.qfrc_actuator.flat,
                data.cfrc_ext.flat,
            ]
        )

    def _get_state(self):
        return self._get_obs()

    def step(self, a):
        # print("self.frame_skip = ", self.frame_skip)
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return (
            self._get_obs(),
            reward,
            done,
            dict(
                reward_linvel=lin_vel_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_alive=alive_bonus,
                reward_impact=-quad_impact_cost,
            ),
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(
                low=-c,
                high=c,
                size=self.model.nv,
            ),
        )
        return self._get_obs()

    def reset(self):
        self.set_state(
            self.init_qpos,
            self.init_qvel,
        )
        return self._get_state()

    # get simulation state
    def get_inner_state(self):
        return self.sim.get_state()

    # set simulation state (more than just the position and velocity) to saved one
    def set_inner_state(self, saved_state):
        self.sim.set_state(saved_state)
        #return self.sim.get_state()
        return self._get_state()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20


class MyHumanoidEnv:
    TARGET_SHAPE = 0
    MAX_PIX_VALUE = 0

    def __init__(self):
        self.env = HumanoidEnv()
        self.rooms = []

        self.reset()

    def __getattr__(self, e):
        assert self.env is not self
        return getattr(self.env, e)

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self.env._get_state()

    def get_restore(self):
        return (
            self.env.get_inner_state(),
        )

    def restore(self, data):
        self.env.set_inner_state(data[0])
        return self.env._get_state()

    def step(self, action):
        return self.env.step(action)

    def get_state(self):
        #return self.env._get_state()
        return self.env._get_state()

    def render_with_known(self, known_positions, resolution, show=True, filename=None, combine_val=max,
                          get_val=lambda x: x.score, minmax=None):
        pass
