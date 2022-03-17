from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
import torch.utils.data as Data
from torch.autograd import Variable

if th.cuda.is_available():
  device = th.device("cuda")
else:
  device = th.device("cpu")

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps


from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
# from SAC_utils.vec_normalize_fetch import VecNormalize

import copy


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        # L_goals,
        # L_steps,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 1000,
        batch_size: int = 256,
        tau: float = 0.005, #0.001, #0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        # ent_coef: Union[str, float] = "auto",
        ent_coef: Union[str, float] = 0.001,
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        warmup_duration: int = 100,
        verbose: int = 0,
        seed: Optional[int] = None,
        make_logs = False,
        path ="",
        bc_reg_bool = False,
        bonus_reward_bool = True,
        add_ent_reg_critic = True,
        alpha_bonus = 1.,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(SAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        self.add_ent_reg_critic = add_ent_reg_critic

        self.update_freq = 100
        self.train_iteration = 0
        self.all_skills_feasible = False

        ## params for reward bonus
        self.eps_tolerance = 0.03 ## for relabelling close to actual goals
        self.add_bonus_reward = bonus_reward_bool
        self.warmup_duration = learning_starts ## number of training steps before adding the bonus
        self.alpha_bonus = alpha_bonus
        self.max_reward = self.env.envs[0].max_reward

        ## extract _vec_normalize_env
        # self._vec_normalize_env = self.env
        # print("self._vec_normalize_env = ", self._vec_normalize_env)

        ## log for critic divergence analysis
        self.log_losses_freq = 100

        self.make_logs = make_logs
        ## save critics value for further analysis
        if self.make_logs:
            ## save critics value for further analysis
            self.f_log_next_q_values = open(path + "/log_next_q_values.txt", "w")
            self.f_log_target_q_values = open(path + "/log_target_q_values.txt", "w")
            self.f_next_log_prob = open(path + "/log_next_log_prob.txt", "w")
            self.f_log_current_q_values = open(path + "/log_current_q_values.txt", "w")
            self.f_weights_sum = open(path + "/log_weights_sum.txt", "w")

            ## save critic and actor losses for further analysis
            self.f_critic_losses = open(path + "/critic_losses.txt", "w")
            self.f_actor_losses = open(path + "/actor_losses.txt", "w")

            self.relabel_success_error = 0
            self.relabel_bs_error = 0
            self.success_error = 0
            self.bs_error = 0

            self.f_critic_errors = open(path + "/critic_errors.txt", "w")

        if _init_setup_model:
            self._setup_model()

        ## SAC part
    def _setup_model(self) -> None:
        super(SAC, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))

        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)


    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.train_iteration += 1

        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Atomic skill feasibility:
        #if not self.all_skills_feasible :
            #print("check feasible ", self.all_skills_feasible)
            #self.all_skills_feasible = all(skill_feasible for skill_feasible in self.env.envs[0].skill_manager.L_skills_feasible[1:])
            #for g in self.critic.optimizer.param_groups:
                #print("g['lr'] = ", g['lr'])
            ## decrease lr to improve stability after adding exploratory bonus
            #if self.all_skills_feasible:
                #self.replay_buffer.n_sampled_goal = 1
                #self.replay_buffer.her_ratio = 1 - (1.0 / (self.replay_buffer.n_sampled_goal + 1))


        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # replay_data, infos =  self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # replay_data, infos, n_step_returns, n_step_observations =  self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            replay_data, infos, next_ep_final_transitions_infos, her_indices, _, _, _, _, _, _, _ =  self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            ## R \in {0,1}
            diff_reward_done = replay_data.dones - replay_data.rewards/self.max_reward
            dones = copy.deepcopy(replay_data.dones)

            ## add dones for relabelled transitions
            missing_dones = (diff_reward_done < 0).int()

            dones[:] = dones[:] + missing_dones[:]
            assert (dones[:] <= 1).all()
            assert (dones[:] >= 0).all()

            ## remove dones for timeouts
            # extra_dones = (diff_reward_done > 0).int()
            # replay_data.dones[:] = replay_data.dones[:] - extra_dones[:]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            ## Transform rewards only if warm-up phase for Q-value done.
            if self.train_iteration > self.warmup_duration and self.add_bonus_reward:# and self.all_skills_feasible:

                ## value-based bonus reward
                overshoot_goal = []
                desired_goals = []
                next_desired_goals = []
                for info in infos:
                    desired_goals.append(list(self.env.envs[0].project_to_goal_space(self.env.envs[0].skill_manager.get_goal_state(info[0]["goal_indx"]))))

                    next_goal_indx = info[0]["goal_indx"]+1
                    if next_goal_indx < len(self.env.envs[0].skill_manager.L_states):
                        next_desired_goals.append(list(self.env.envs[0].project_to_goal_space(self.env.envs[0].skill_manager.get_goal_state(next_goal_indx))))
                    else:
                        next_desired_goals.append(list(self.env.envs[0].project_to_goal_space(self.env.envs[0].skill_manager.get_goal_state(next_goal_indx-1))))

                    if info[0]["goal_indx"] < len(self.env.envs[0].skill_manager.L_states)-1: ## add bonus to subgoals only (not final goal)
                        overshoot_goal.append([1])
                    else:
                        overshoot_goal.append([0])


                transformed_rewards = self._transform_rewards(replay_data,  infos, desired_goals, next_desired_goals, her_indices, overshoot_goal)
                transformed_rewards = transformed_rewards.detach()

                assert len(infos) == len(next_ep_final_transitions_infos)

            else:
                transformed_rewards = replay_data.rewards

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            # print("self.target_entropy = ", self.target_entropy)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)

                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                # add entropy term
                next_q_values = next_q_values - self.add_ent_reg_critic * ent_coef * next_log_prob.reshape(-1, 1)

                # td error + entropy term
                target_q_values = (transformed_rewards + (1 - dones) * self.gamma * next_q_values).float()

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            for i in range(current_q_values[0].shape[0]):
                if abs(current_q_values[0][i] - target_q_values[i]) > 1.:
                    #print("\n diff values = ", abs(current_q_values[0][i] - target_q_values[i]))
                    #print("current value = ", current_q_values[0][i])
                    #print("target value = ",  target_q_values[i])
                    #print("transformed reward = ", transformed_rewards[i])
                    #print("dones = ", dones[i])
                    #print("reward = ", replay_data.rewards[i])
                    #print("i = ", i)

                    ## relabelling
                    if i < int(self.replay_buffer.her_ratio*self.batch_size):
                        if dones[i] == 0.:
                            self.relabel_bs_error+= 1
                        else:
                            self.relabel_success_error+=1
                    ## no relabelling
                    else :
                        if dones[i] == 0.:
                            self.bs_error+= 1
                        else:
                            self.success_error+=1



            ## check weights values
            L_weights = [self.critic.q_networks[0][0].weight.flatten(), self.critic.q_networks[0][2].weight.flatten(), self.critic.q_networks[0][4].weight.flatten()]
            cat_weights = th.cat((L_weights), axis=-1)
            sum_weights = cat_weights.sum()

            # Compute critic loss (mean error over 2 critics)
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])

            critic_losses.append(critic_loss.item())


            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            #q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            q_values_pi = th.cat(self.critic_target.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()

            actor_loss.backward()

            self.actor.optimizer.step() ## optimize actor

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        if self.train_iteration % self.log_losses_freq == 0 and self.make_logs:
            self.f_log_next_q_values.write(str(next_q_values.mean()) + "\n")
            self.f_log_next_q_values.flush()
            # print("target_q_values[0] = ", target_q_values[0])
            self.f_log_target_q_values.write(str(target_q_values.mean()) + "\n")
            self.f_log_target_q_values.flush()
            self.f_next_log_prob.write(str(next_log_prob.mean()) + "\n")
            self.f_next_log_prob.flush()
            self.f_log_current_q_values.write(str(current_q_values[0].mean()) + "\n")
            self.f_log_current_q_values.flush()
            self.f_critic_losses.write(str(np.mean(critic_losses)) + "\n")
            self.f_critic_losses.flush()
            self.f_actor_losses.write(str(np.mean(actor_losses)) + "\n")
            self.f_actor_losses.flush()
            self.f_weights_sum.write(str(sum_weights) + "\n")
            self.f_weights_sum.flush()
            self.f_critic_errors.write(str(self.relabel_bs_error) + " " +
                                    str(self.relabel_success_error) + " " +
                                    str(self.bs_error) + " " +
                                    str(self.success_error) + "\n")
            self.f_critic_errors.flush()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))




    def _transform_rewards(self, replay_data, infos, true_desired_goals, shift_desired_goals, her_indices, overshoot_goal):

        rewards = copy.deepcopy(replay_data.rewards)

        next_observations = copy.deepcopy(replay_data.next_observations) ## includes n-step observation

        # print("next observation = ", replay_data.next_observations["observation"][0])
        # print("achieved_goal = ", replay_data.next_observations["achieved_goal"][0])
        ## shift desired goal to compute reward bonus
        next_observations_shift_desired_goal = copy.deepcopy(replay_data.next_observations)
        next_observations_shift_desired_goal["observation"] = next_observations_shift_desired_goal["observation"].cpu().numpy()
        next_observations_shift_desired_goal["achieved_goal"] = next_observations_shift_desired_goal["achieved_goal"].cpu().numpy()
        next_observations_shift_desired_goal["desired_goal"] = next_observations_shift_desired_goal["desired_goal"].cpu().numpy()
        next_observations_shift_desired_goal = self._vec_normalize_env.unnormalize_obs(next_observations_shift_desired_goal)
        next_observations_shift_desired_goal["desired_goal"][:] = np.array(shift_desired_goals)[:]
        next_observations_shift_desired_goal = self._vec_normalize_env.normalize_obs(next_observations_shift_desired_goal)
        next_observations_shift_desired_goal["observation"] = th.from_numpy(next_observations_shift_desired_goal["observation"]).to(self.device)
        next_observations_shift_desired_goal["achieved_goal"] = th.from_numpy(next_observations_shift_desired_goal["achieved_goal"]).to(self.device)
        next_observations_shift_desired_goal["desired_goal"] = th.from_numpy(next_observations_shift_desired_goal["desired_goal"]).to(self.device)

        # Get next action using current policy
        next_actions = self.actor._predict(next_observations_shift_desired_goal, deterministic=True)

        # Compute the next Q values
        next_values = th.cat(self.critic_target.forward(next_observations_shift_desired_goal, next_actions), dim=1)
        next_values = th.min(next_values, dim=1, keepdim=True)[0].detach()

        # update chaining bonus reward if larger than current value
        #chaining_bonus_reward = []
        # print("\nnext_values.shape = ", next_values.shape)
        # print("next_values[:10] = ", next_values[:10])

        #for i in range(len(infos)):
            #if next_values[i] > infos[i][0]["chaining_bonus_reward"]:
                #infos[i][0]["chaining_bonus_reward"] = next_values[i]
            #chaining_bonus_reward.append(list(infos[i][0]["chaining_bonus_reward"].cpu().numpy()))
        #t_chaining_bonus_reward = th.FloatTensor(chaining_bonus_reward).to(self.device)
        # print("t_chaining_bonus_reward.shape = ", t_chaining_bonus_reward.shape)
        # print("t_chaining_bonus_reward[:10] = ", t_chaining_bonus_reward[:10])

        # Compute reward mask -> only success can receive bonus reward
        assert (rewards <= self.max_reward).all()
        success_mask = (rewards == self.max_reward).int() ## includes n-step returns

        ## last goal should not look for overshoot success
        goal_mask = np.array(overshoot_goal)

        ## no bonus for relabelled transitions (except if relabelled desired goal close to actual desired goal?)
        next_observations["observation"] = next_observations["observation"].cpu()
        next_observations["achieved_goal"] = next_observations["achieved_goal"].cpu()
        next_observations["desired_goal"] = next_observations["desired_goal"].cpu()
        next_observations = self._vec_normalize_env.unnormalize_obs(next_observations)
        relabelling_mask = (th.linalg.norm(next_observations["desired_goal"][:] - th.FloatTensor(true_desired_goals)[:], axis=1) < 0.01).float().reshape(rewards.shape)

        # Compute reward bonus by successive application of reward and total dist masks
        # reward_bonus = next_values * success_mask.float() * th.from_numpy(relabelling_mask).float().to(self.device) * th.from_numpy(goal_mask).float().to(self.device)
        reward_bonus = next_values * success_mask.float() * relabelling_mask.to(self.device) * th.from_numpy(goal_mask).float().to(self.device)
        #reward_bonus = t_chaining_bonus_reward * success_mask.float() * relabelling_mask.to(self.device) * th.from_numpy(goal_mask).float().to(self.device)

        return rewards + reward_bonus.float()


    def _sample_buffer(self,
        batch_size):
        """
        Sample replay buffer without relabelling
        """

        buffer = self.replay_buffer._buffer
        ## sample episodes indices
        if self.replay_buffer.full:
            episode_indices = (np.random.randint(1, self.replay_buffer.n_episodes_stored, batch_size) + self.replay_buffer.pos) % self.replay_buffer.n_episodes_stored
        else:
            episode_indices = np.random.randint(0, self.replay_buffer.n_episodes_stored, batch_size)

        ## gather values from replay_buffer class
        ep_lengths = self.replay_buffer.episode_lengths[episode_indices]
        transitions_indices = np.random.randint(ep_lengths)
        transitions = {key: self.replay_buffer._buffer[key][episode_indices, transitions_indices].copy() for key in self.replay_buffer._buffer.keys()}

        return transitions


    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "SAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(SAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
