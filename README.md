# Divide & Conquer Imitation Algorithm (NOT TESTED YET)


### Requirements

- stablebaselines3
- pytorch
- gym 
- matplotlib
- pickle
- numpy
- cv2
- seaborn
- tensorboard (for SAC update monitoring)


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
- evaluate/ contains the function to evaluate the learned Goal-Conditioned Policy on the chaining of tasks,
- demos/ contains the demonstration for each environment,
- demo_extractors/ contains the class used to extract the demonstration and prepare the environment,
- callbacks/ contains custom Callback classes (for SB3) used to obtain visual logs 
- envs/ contains two environments : DubinsMazeEnv and FetchEnv
- xp/ contains the saved logs of the launched experiments (note that additional logs like critic loss, actor loss, etc. can be displayed in the tensorboard if necessary). 

### Launch commands


For the DubinsMaze environment: 
```sh
python run_DCIL_maze.py --eps_dist 1. --algo SAC_HER --env DubinsMazeEnv --size 5 --demo_path ./demos/dubinsmazeenv/demo_set_5/ --bonus_bool 1 --overshoot_bool 1 -x 1
```

For the FetchEnv environment
```sh
python run_DCIL_fetch.py --eps_dist 0.5 --algo SAC_HER --env FetchEnv_grasping -l 0.001 --demo_path ./demos/fetchenv/demo_set/ --bonus_bool 1 --overshoot_bool 1 -x 1 --eps_optimizer 0.001
```

### How does it work
Paper available soon

### Understanding the visual logs (iteration_*it*.png)
TODO 

### Training example 

DubinsMaze environment: 

![](https://github.com/AlexandreChenu/DCIL/media/DCIL_dubins1.gif)
