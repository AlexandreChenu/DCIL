import gym
from gym.envs.registration import register

# simple maze environment
from .simple_mazeenv.mazeenv_cst_speed_wrappers import SimpleMazeEnvGCPHERSB3

# dubins maze environment
from .dubins_mazeenv.mazeenv_cst_speed_wrappers import DubinsMazeEnvGCPHERSB3

# fetch environment from Go-Explore 2 (Ecoffet et al.)
from .fetchenv.fetch_env import MyComplexFetchEnv
from .fetchenv.fetchenv_wrappers import ComplexFetchEnvGCPHERSB3

print("REGISTERING SimpleMazeEnv")
register(
    id='SimpleMazeEnvGCPHERSB3-v0',
    # entry_point='envs.dubins_mazeenv.mazeenv_wrappers:DubinsMazeEnvGCPHERSB3')
    entry_point='envs.simple_mazeenv.mazeenv_cst_speed_wrappers:SimpleMazeEnvGCPHERSB3')

## mazeenv from Guillaume Matheron with a Dubins car
print("REGISTERING DubinsMazeEnv")
register(
    id='DubinsMazeEnvGCPHERSB3-v0',
    # entry_point='envs.dubins_mazeenv.mazeenv_wrappers:DubinsMazeEnvGCPHERSB3')
    entry_point='envs.dubins_mazeenv.mazeenv_cst_speed_wrappers:DubinsMazeEnvGCPHERSB3')


## fetch environment from Go-Explore 2 (Ecoffet et al.)
print("REGISTERING FetchEnv-v0 & FetchEnvGCPHERSB3-v0")
register(
    id='FetchEnv-v0',
    entry_point='envs.fetchenv.fetch_env:MyComplexFetchEnv',
)

register(
    id='FetchEnvGCPHERSB3-v0',
    entry_point='envs.fetchenv.fetchenv_wrappers:ComplexFetchEnvGCPHERSB3',
)

## Humanoid environment from Mujoco
print("REGISTERING HumanoidEnv-v0 & HumanoidEnvGCPHERSB3-v0")
register(
    id='HumanoidEnv-v0',
    entry_point='envs.humanoid.humanoidenv:MyHumanoidEnv',
)

register(
    id='HumanoidEnvGCPHERSB3-v0',
    entry_point='envs.humanoid.humanoidenv_wrappers:HumanoidEnvGCPHERSB3',
)
