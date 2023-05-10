import argparse
import time

import numpy as np
import stable_baselines3 as sb3
import torch as th

from flexible_exp_manager import FlexibleExperimentManager
from rl_zoo3.utils import ALGOS, StoreDict
from cont_drm.cont_drm import C_DRM
from disc_drm.disc_drm import D_DRM

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from setup_training_utils import add_other_parser_args, env_seed_uuid_setup

def setup_and_run_parser(parser):
    #~~~ ALGORITHM SPECIFIC ARGUMENTS ~~~#
    default_algo = "cont_drm"
    parser.add_argument("--algo", help="RL Algorithm", default=default_algo, type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--n-critics",help="Number of critics in ensemble", default=2, type=int)
    parser.add_argument("--use-shaping", help="Flag determining whether to use reward shaping", default=1, type=int)
    parser.add_argument("--use-shaping-scaling", help="Flag determining whether to use DRM scaling on reward shaping", default=1, type=int)

    #~~~ RUN SPECIFIC ARGUMENTS ~~~#
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0", help="environment ID")

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
        default=f"{default_algo}/{default_algo}.yml",
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
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="logs")
    parser.add_argument("--wandb-project-name", type=str, default="DRM MtnCar", help="the wandb's project name")
    parser.add_argument("--wandb-run-name",type=str, default=None, help="the run name to be used in wandb")

    args = parser.parse_args()

    #set default wandb run name if necessary
    if args.wandb_run_name != None: pass
    else: args.wandb_run_name = f"{args.env}__{args.algo}__{args.seed}__{int(time.time())}"

    #set default config file if necessary
    if args.conf_file != None: pass
    else: args.conf_file = f"{args.algo}/{args.algo}.yml"

    return args

def train() -> None:
    #~~~~~~~~~ SETUP PARSER~~~~~~~~~#
    parser = argparse.ArgumentParser()
    parser = add_other_parser_args(parser) #add irrelevant args

    args = setup_and_run_parser(parser)

    #~~~~~~~~~ SETUP ENVIRONMENT, SEED, UUID ~~~~~~~~~#
    args, env_id, uuid_str = env_seed_uuid_setup(args)

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
        args.tensorboard_log = f"runs/{run_name}"

    #Create new algorithm list which includes DRM
    FULL_ALGO_LIST = ALGOS
    FULL_ALGO_LIST["cont_drm"] = C_DRM
    FULL_ALGO_LIST["disc_drm"] = D_DRM

    #initialize policy kwargs with n_critics
    policy_kwargs = {}
    policy_kwargs['n_critics'] = args.n_critics

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
        use_shaping=args.use_shaping,
        use_shaping_scaling=args.use_shaping_scaling,
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

if __name__ == "__main__":
    train()