# Divide & Conquer Imitation Algorithm 


### Requirements

- stablebaselines3
- pytorch
- gym 
- matplotlib
- pickle
- numpy
- cv2
- seaborn
- tensorboard 

Note: versions detailed in dcil_env.yml

### Dependencies for Fetch environment

To run DCIL in the fetch environment, please clone Go-Explore repo:

```sh
git clone https://github.com/uber-research/go-explore.git
```

move to the robustified directory: 

```sh
cd go-explore/robustified
```

From there, clone uber-research/atari-reset: 

```sh
git clone https://github.com/uber-research/atari-reset.git atari_reset
```

Finally, update the PYTHONPATH to include robustified: 

```sh
export PYTHONPATH=$PYTHONPATH:path_to_goexplore/go-explore/robustified
```

with path_to_goexplore, the absolute path to the go-explore repo. 


### Code structure

- run_DCIL_*envname*.py : main file for launching DCIL,
- SAC_utils/ contains the variant of SAC including chaining bonus of reward,
- evaluate/ contains the function to evaluate the skill-chaining performed by the learned Goal-Conditioned Policy,
- demos/ contains several demonstrations for each environment,
- demo_extractors/ contains the class used to extract the demonstration and prepare the environment,
- callbacks/ contains custom Callback classes (Stable-baselines3) to obtain visual logs 
- envs/ contains the environments : DubinsMazeEnv (ours), FetchEnv (First return then explore paper) & Humanoid (Mujoco)
- xp/ contains the saved logs of the launched experiments (note that additional logs like critic loss, actor loss, etc. can be monitored using tensorboard if necessary). 

### Launch commands

For the toy DubinsMaze environment | Tested: successful skill-chaining after ~15k training steps: 

```sh
python run_DCIL_toy_maze.py --algo SAC_HER --add_bonus 1 --add_ent_reg 0
```

For the DubinsMaze environment: 
```sh
python run_DCIL_maze.py --eps_dist 1. --algo SAC_HER --env DubinsMazeEnv --size 5 --demo_path ./demos/dubinsmazeenv/demo_set5/ --bonus_bool 1 --overshoot_bool 1 --add_ent_reg 0 -x 1
```

For the FetchEnv environment
```sh
python run_DCIL_fetch.py --eps_dist 0.5 --algo SAC_HER --env FetchEnv_full -l 0.001 --demo_path ./demos/fetchenv/ --bonus_bool 1 --overshoot_bool 1 --ent_reg_bool 0 -x 1
```

### How does it work
Paper available soon

### Training + skill-chaining Dubins Maze

![](https://github.com/AlexandreChenu/DCIL/blob/main/media/DCIL_dubins1.gif)

### Skill-chaining Fetch 

![](https://github.com/AlexandreChenu/DCIL/blob/main/media/DCIL_fetch.gif)
