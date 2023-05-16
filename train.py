import argparse
import time

import numpy as np
import stable_baselines3 as sb3
import torch as th

from flexible_exp_manager import FlexibleExperimentManager, LakeRewardWrapper, LAKE_DESC
from rl_zoo3.utils import ALGOS, StoreDict
from cont_drm.cont_drm import C_DRM
from disc_drm.disc_drm import D_DRM

import os
import gym

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from setup_training_utils import add_other_parser_args, env_seed_uuid_setup

from shaping_functions.car_reward_shaping import car_reward_shaping
from shaping_functions.lunar_lander_reward_shaping import lander_reward_shaping
from shaping_functions.lake_reward_shaping import lake_reward_shaping
from shaping_functions.cliff_reward_shaping import cliff_reward_shaping

#Create new algorithm list which includes DRM
FULL_ALGO_LIST = ALGOS
FULL_ALGO_LIST["cont_drm"] = C_DRM
FULL_ALGO_LIST["disc_drm"] = D_DRM

SHAPING_SCALING_TYPES = ["count", "drm", "naive", "rnd"]
ENV_SHAPING_FUNCTIONS = {"CliffWalking-v0":cliff_reward_shaping, "FrozenLake-v1":lake_reward_shaping, \
                         "LunarLander-v2":lander_reward_shaping, "MountainCarContinuous-v0":car_reward_shaping}

def setup_and_run_parser(parser):
    #~~~ ALGORITHM SPECIFIC ARGUMENTS ~~~#
    parser.add_argument("--algo", help="RL Algorithm", default="disc_drm", type=str, required=False, choices=list(FULL_ALGO_LIST.keys()))
    parser.add_argument("--n-qnets",help="Number of Q-networks in ensemble", default=2, type=int)
    parser.add_argument("--scaling", help="Shaping scaling type", default=None, type=str, required=False, choices=SHAPING_SCALING_TYPES)
    parser.add_argument("--no-shaping", action="store_true", help="Flag determining whether to use reward shaping.", default=False)

    #~~~ RUN SPECIFIC ARGUMENTS ~~~#
    parser.add_argument("--env", type=str, default="FrozenLake-v1", help="environment ID")

    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=-1, type=int)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument(
        "-conf",
        "--conf-file",
        type=str,
        default=None,
        help="Custom yaml file or python package from which the hyperparameters will be loaded."
        "We expect that python packages contain a dictionary called 'hyperparams' which contains a key for each environment.",
    )

    #~~~ REPORTING SPECIFIC ARGUMENTS ~~~#
    parser.add_argument(
        "--track",
        action="store_true",
        default=False,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument("--wandb-project-name", type=str, default="Discrete DRM", help="the wandb's project name")
    parser.add_argument("--wandb-run-name",type=str, default=None, help="the run name to be used in wandb")

    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--video", action="store_true", default=False, help="Flag determining whether to save a video at the end of training")

    args = parser.parse_args()

    #set default wandb run name if necessary
    if args.wandb_run_name != None: pass
    else: args.wandb_run_name = f"{args.env}__{args.algo}__{args.seed}__{int(time.time())}"

    #set default config file if necessary
    if args.conf_file == None: args.conf_file = f"{args.algo}/{args.algo}.yml"
    else: pass

    return args

def train() -> None:
    #~~~~~~~~~ SETUP PARSER~~~~~~~~~#
    parser = argparse.ArgumentParser()
    parser = add_other_parser_args(parser) #add irrelevant args

    args = setup_and_run_parser(parser)

    #~~~~~~~~~ SETUP ENVIRONMENT, SEED, UUID ~~~~~~~~~#
    args, env_id, uuid_str = env_seed_uuid_setup(args)

    #~~~~~~~~~ SETUP REWARD SHAPING ~~~~~~~~~#
    if args.no_shaping: shaping_function = None
    else: 
        try: shaping_function = ENV_SHAPING_FUNCTIONS[env_id]
        except KeyError: raise KeyError(f"Environment {env_id} does not have a reward shaping function implemented.")

    #~~~~~~~~~ SETUP W&B ~~~~~~~~~#
    if args.track:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            )

        tags = args.wandb_tags + [f"v{sb3.__version__}"]
        run = wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            tags=tags,
            config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        args.tensorboard_log = f"runs/{args.wandb_run_name}"

    #initialize policy kwargs with algorithm specific arguments
    policy_kwargs = {}
    policy_kwargs['n_qnets'] = args.n_qnets

    exp_manager = FlexibleExperimentManager(
        args,
        args.algo,
        FULL_ALGO_LIST,
        env_id,
        args.log_folder,
        args.tensorboard_log,
        args.n_timesteps,
        args.eval_freq,
        args.eval_episodes,
        args.save_freq,
        args.hyperparams,
        args.env_kwargs,
        args.trained_agent,
        args.optimize_hyperparameters,
        args.storage,
        args.study_name,
        args.n_trials,
        args.max_total_trials,
        args.n_jobs,
        args.sampler,
        args.pruner,
        args.optimization_log_path,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        truncate_last_trajectory=args.truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
        n_eval_envs=args.n_eval_envs,
        no_optim_plots=args.no_optim_plots,
        device=args.device,
        config=args.conf_file,
        show_progress=args.progress,
        policy_kwargs=policy_kwargs,
        shaping_function=shaping_function,
        shaping_scaling_type=args.scaling,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if args.track:
            # we need to save the loaded hyperparameters
            args.saved_hyperparams = saved_hyperparams
            assert run is not None  # make mypy happy
            run.config.setdefaults(vars(args))

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()

    #SAVE VIDEOS (only for discrete environments)
    if args.video and args.algo == "disc_drm":
        vid_folder = os.path.join(args.log_folder, "videos")
        os.makedirs(vid_folder, exist_ok=True)
        video_length = 100

        vec_env = DummyVecEnv([lambda: LakeRewardWrapper(gym.make(env_id,desc=LAKE_DESC,is_slippery=False),-0.01,0)])

        # Record the video starting at the first step
        vec_env = VecVideoRecorder(vec_env, vid_folder,
                            record_video_trigger=lambda x: x == 0, video_length=video_length,
                            name_prefix=f"{args.wandb_run_name}_vid")

        obs = vec_env.reset() 


        vec_env.reset()
        for _ in range(video_length + 1):
            action = model.policy.predict(obs)
            obs, _, _, _ = vec_env.step(action[0])
        # Save the video
        vec_env.close()

if __name__ == "__main__":
    train()