import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update, should_collect_more_steps
from disc_drm.disc_drm_policies import CnnPolicy, D_DRMPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from scaling_functions.count_reward_scaling import countbased_reward_scaling
from scaling_functions.rnd_reward_scaling import RunningMeanStd, RewardForwardFilter

from collections import defaultdict

SelfD_DRM = TypeVar("SelfD_DRM", bound="D_DRM")

class D_DRM(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[D_DRMPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        shaping_starts: int = 0,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        n_qnets: int = 2,
        shaping_function = None,
        shaping_scaling_type: str = None,
        count_temp: int = 1,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_nets, self.q_net_targets = None, None

        self.shaping_function = shaping_function
        self.shaping_scaling_type = shaping_scaling_type

        self.max_avg_batch_q_stds = 0.0001 #start at non-zero value to prevent divide by zero
        self.max_avg_rnd_diff = 0.00001

        self.n_qnets = n_qnets

        self.shaping_starts = shaping_starts

        self.count_temp = count_temp
        self.state_counts = defaultdict(int)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_nets, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_targets, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

        if self.shaping_scaling_type == "rnd":
            #Initialize RND data collection for observation normalization:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)

            starting_observations = []
            self.env.reset()
            for step in range(10000):
                actions = np.array([self.action_space.sample() for _ in range(self.n_envs)])
                
                obs, rewards, dones, infos = self.env.step(actions)
                
                

    def _create_aliases(self) -> None:
        self.q_nets = self.policy.q_nets
        self.q_net_targets = self.policy.q_net_targets
        self.rnd_target = self.policy.rnd_target
        self.rnd_learner = self.policy.rnd_learner

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_nets.parameters(), self.q_net_targets.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate([self.policy.optimizer])

        losses, q_stds_for_all_batches, reward_scalings, rnd_losses = [], [], [], []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Get current Q-values estimates
            current_q_values = self.q_nets(replay_data.observations)
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = [th.gather(current_q_val, dim=1, index=replay_data.actions.long()) for current_q_val in current_q_values]

            with th.no_grad():
                # # #REGULAR DQN UPDATES
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_targets(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                greedy_next_qvals, greedy_next_actions = [], []
                for next_q_val in next_q_values:
                    greedy_next_qval, greedy_next_action = next_q_val.max(dim=1)
                    greedy_next_qvals.append(greedy_next_qval.unsqueeze(1))
                    greedy_next_actions.append(greedy_next_action.unsqueeze(1))

                greedy_next_qvals = th.cat(greedy_next_qvals, dim=1)

                #Get current q values from the current network to use to calculate variance
                next_q_values_from_current = self.q_nets(replay_data.next_observations) #Q values for all actions from all ensembles
                best_next_q_values_from_current = [] #will grab only the Q values for the greedy actions from all ensembles
                for next_q_val_from_current, greedy_next_action in zip(next_q_values_from_current, greedy_next_actions):
                    best_next_q_values_from_current.append(th.gather(next_q_val_from_current, dim=1, index=greedy_next_action.long()))

                #~~~~Calculate q function variance over time. Calculate this even without
                #shaping scaling for the sake of logging~~~~~

                #Add the Q values for the greedy actions from all ensembles into a single tensor so we can take std
                single_tensor_curr_q_values = th.cat(current_q_values, dim=1)
                q_stds_for_batch = th.std(single_tensor_curr_q_values, dim=1, correction=0, keepdim=True)
                avg_batch_q_diff = th.mean(q_stds_for_batch).item()
                q_stds_for_all_batches.append(avg_batch_q_diff)

                if avg_batch_q_diff > self.max_avg_batch_q_stds:
                    self.max_avg_batch_q_stds = avg_batch_q_diff

                #################### Calculate appropriate reward shaping and scaling ################
                if self.shaping_function != None:
                    shaping_rewards = self.shaping_function(replay_data.observations, replay_data.actions)

                    if self.shaping_scaling_type == None:
                        shaping_reward_scaling = th.ones_like(shaping_rewards,dtype=th.float16)

                    elif self.shaping_scaling_type == "count":
                        shaping_reward_scaling = countbased_reward_scaling(self.state_counts, self.count_temp, replay_data.observations, replay_data.actions)
                        shaping_reward_scaling = th.minimum(shaping_reward_scaling, th.ones_like(shaping_reward_scaling))

                    elif self.shaping_scaling_type == "drm":
                        shaping_reward_scaling = th.minimum(q_stds_for_batch/self.max_avg_batch_q_stds, th.ones_like(q_stds_for_batch))

                    elif self.shaping_scaling_type == "rnd":
                        target_rnd_vals = self.rnd_target(replay_data.observations)

                        rnd_differences = th.abs(target_rnd_vals - self.rnd_learner(replay_data.observations))
                        avg_rnd_differences = rnd_differences.mean().item()

                        if avg_rnd_differences > self.max_avg_rnd_diff:
                            self.max_avg_rnd_diff = avg_rnd_differences
                        
                        normalized_rnd_differences = rnd_differences/self.max_avg_rnd_diff

                        shaping_reward_scaling = th.minimum(normalized_rnd_differences, th.ones_like(normalized_rnd_differences))
                    
                    elif self.shaping_scaling_type == "naive":
                        shaping_reward_scaling = self._current_progress_remaining*th.ones_like(q_stds_for_batch)
                    
                    reward_scalings.append(shaping_reward_scaling.mean().item())
                    
                    shaping_rewards = th.mul(shaping_rewards, shaping_reward_scaling)

                else: #if no shaping function provided, just use the regular rewards
                    shaping_reward_scaling = th.zeros_like(replay_data.rewards)
                    shaping_rewards = th.zeros_like(replay_data.rewards)

                target_q_values = replay_data.rewards + shaping_rewards + (1 - replay_data.dones) * self.gamma * greedy_next_qvals

                reward_scalings.append(shaping_reward_scaling.mean().item())

            # Compute Huber loss (less sensitive to outliers)
            losses_for_ensemble = [F.smooth_l1_loss(current_q_val, target_q_values[:,i].unsqueeze(1)) for i, current_q_val in enumerate(current_q_values)]
            losses.append((sum(losses_for_ensemble)/len(losses_for_ensemble)).item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            for loss in losses_for_ensemble:
                loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            if self.shaping_scaling_type == "rnd" and self.shaping_function != None:
                #Compute RND loss
                rnd_loss = F.mse_loss(self.rnd_learner(replay_data.observations), target_rnd_vals)
                rnd_losses.append(rnd_loss.item())

                #optimize the RND learner
                self.rnd_learner.optimizer.zero_grad()
                rnd_loss.backward()
                th.nn.utils.clip_grad_norm_(self.rnd_learner.parameters(), self.max_grad_norm)
                self.rnd_learner.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        print(f"{self._n_calls}",end="\r")

        # self.logger.record("train/max_avg_rnd_std", self.max_avg_rnd_loss)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/qstds", np.mean(q_stds_for_all_batches))
        if self.shaping_scaling_type == "rnd" and self.shaping_function != None: 
            self.logger.record("train/max_avg_rnd_std", self.max_avg_rnd_diff)
            self.logger.record("train/rnd_losses", np.mean(rnd_losses))
        else: self.logger.record("train/max_qstds", self.max_avg_batch_q_stds)

        if self.shaping_function != None: self.logger.record("train/reward_scale", np.mean(reward_scalings))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic, q_net_index_to_use=self.q_net_index_to_use)
        return action, state

    def learn(
        self: SelfD_DRM,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "D_DRM",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfD_DRM:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["q_nets", "q_net_targets", "state_counts"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        NOTE: Almost identical to standard collect_rollouts, except at the start of a rollout we need
        to randomly pick a network to select observations from, and stick with it for the entire rollout.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        self.q_net_index_to_use = np.random.choice(self.n_qnets)

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            self.state_counts[(self._last_obs[0],actions[0])] += 1 #grab the only element from the numpy array

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1
                    self.q_net_to_use = np.random.choice(self.n_qnets) #update q network to use

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)