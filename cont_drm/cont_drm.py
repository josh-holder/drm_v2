from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update,should_collect_more_steps
from cont_drm.cont_drm_policies import C_DRMPolicy, CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from scaling_functions.count_reward_scaling import countbased_reward_scaling
from scaling_functions.rnd_reward_scaling import RunningMeanStd, RewardForwardFilter

SelfC_DRM = TypeVar("SelfC_DRM", bound="C_DRM")

class C_DRM(OffPolicyAlgorithm):
    """
    Dynamic Reward Modification (C_DRM)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Based on stable_baselines TD3 implementation.

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
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
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
        policy: Union[str, Type[C_DRMPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        shaping_function = None,
        shaping_scaling_type: str = None,
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
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.shaping_function = shaping_function
        self.shaping_scaling_type = shaping_scaling_type

        self.max_avg_batch_q_stds = 0.00001
        self.max_avg_rnd_diff = 0.00001

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        if self.shaping_scaling_type == "rnd":
            #Initialize RND data collection for observation normalization:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
            self.int_rew_rms = RunningMeanStd(shape=(1,))

            starting_observations = []
            self.env.reset()
            for step in range(10000):
                actions = np.array([self.action_space.sample() for _ in range(self.n_envs)])
                
                obs, _, _, _ = self.env.step(actions)

                starting_observations.append(obs)
            
            starting_observations = np.stack(starting_observations)
            self.obs_rms.update(starting_observations)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        self.rnd_target = self.policy.rnd_target
        self.rnd_learner = self.policy.rnd_learner

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, q_stds_for_all_batches, reward_scalings, rnd_losses = [], [], [], [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            #(batch_size)x(n) tensor of curr. Q values, rather than tuple of n (batchsize)x1 tensors
            single_tensor_current_q_values = th.cat(current_q_values, dim=1)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)

                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                m = 2
                critic_indices_to_use = np.random.choice(self.critic.n_critics, m, replace=False)

                # Compute the next Q-values: min over a random selection of m of the critics.
                next_q_values = []
                for critic_index in critic_indices_to_use:
                    features = self.critic_target.extract_features(replay_data.next_observations, self.critic_target.features_extractor)
                    next_q_values.append(self.critic_target.q_networks[critic_index](th.cat([features, next_actions], dim=1)))
                    
                next_q_values = th.cat(next_q_values, dim=1) #change tuple to single tensor
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                
                q_stds_for_batch = th.std(single_tensor_current_q_values, dim=1)
                avg_batch_q_std = th.mean(q_stds_for_batch).item()
                q_stds_for_all_batches.append(avg_batch_q_std)

                if avg_batch_q_std > self.max_avg_batch_q_stds:
                    self.max_avg_batch_q_stds = avg_batch_q_std

                if self.shaping_function != None:
                    shaping_rewards = self.shaping_function(replay_data.observations, replay_data.actions)

                    if self.shaping_scaling_type == None:
                        shaping_reward_scaling = th.ones_like(shaping_rewards,dtype=th.float16)

                    elif self.shaping_scaling_type == "count":
                        print("Count based reward shaping no meaningful in continuous setting - use RND instead.")
                        raise NotImplementedError

                    elif self.shaping_scaling_type == "drm":
                        shaping_reward_scaling = th.minimum(q_stds_for_batch/self.max_avg_batch_q_stds, th.ones_like(q_stds_for_batch))

                    elif self.shaping_scaling_type == "rnd":
                        normalized_observations = th.clip((replay_data.observations-self.obs_rms.mean)/np.sqrt(self.obs_rms.var), -5, 5)

                        target_rnd_vals = self.rnd_target(normalized_observations)

                        rnd_differences = th.abs(target_rnd_vals - self.rnd_learner(normalized_observations))
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

                target_q_values = replay_data.rewards + shaping_rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Compute critic loss
            #F is torch functional
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            #Compute RND loss
            if self.shaping_scaling_type == "rnd" and self.shaping_function != None:
                rnd_loss = F.mse_loss(self.rnd_learner(replay_data.observations), target_rnd_vals)
                rnd_losses.append(rnd_loss.item())

                #optimize the RND learner
                self.rnd_learner.optimizer.zero_grad()
                rnd_loss.backward()
                # th.nn.utils.clip_grad_norm_(self.rnd_learner.parameters(), self.max_grad_norm)
                self.rnd_learner.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/qstds", np.mean(q_stds_for_all_batches))
        if self.shaping_scaling_type == "rnd" and self.shaping_function != None: 
            self.logger.record("train/max_avg_rnd_std", self.max_avg_rnd_diff)
            self.logger.record("train/rnd_losses", np.mean(rnd_losses))
        else: self.logger.record("train/max_qstds", self.max_avg_batch_q_stds)

        if self.shaping_function != None: self.logger.record("train/reward_scale", np.mean(reward_scalings))

    def learn(
        self: SelfC_DRM,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "C_DRM",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfC_DRM:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
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

        new_observations = []

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            new_observations.append(new_obs)

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

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        #If RND is active, update the running average of the observation
        new_observations = np.stack(new_observations)
        self.obs_rms.update(new_observations)

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)