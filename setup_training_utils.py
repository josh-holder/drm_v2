import uuid
from stable_baselines3.common.utils import set_random_seed
import os
import torch as th
import difflib
import importlib
from rl_zoo3.utils import StoreDict
import gym
import numpy as np

def add_other_parser_args(parser):
    """
    Adds all the irrelevant parser arguments, which are not needed for DRM
    """
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="", type=str)

    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training", default="", type=str)
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory "
        "in the replay buffer will be truncated after reloading the replay buffer.",
        default=True,
        type=bool,
    )

    parser.add_argument("--vec-env", help="VecEnv type", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--n-trials",
        help="Number of trials for optimizing hyperparameters. "
        "This applies to each optimization runner, not the entire optimization process.",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-total-trials",
        help="Number of (potentially pruned) trials for optimizing hyperparameters. "
        "This applies to the entire optimization process and takes precedence over --n-trials if set.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-optimize", "--optimize-hyperparameters", action="store_true", default=False, help="Run hyperparameters search"
    )
    parser.add_argument(
        "--no-optim-plots", action="store_true", default=False, help="Disable hyperparameter optimization plots"
    )
    parser.add_argument("--n-jobs", help="Number of parallel jobs when optimizing hyperparameters", type=int, default=1)
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument("--n-startup-trials", help="Number of trials before using optuna sampler", type=int, default=10)
    parser.add_argument(
        "--n-evaluations",
        help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization."
        "Default is 1 evaluation per 100k timesteps.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
    )
    parser.add_argument("--study-name", help="Study name for distributed optimization", type=str, default=None)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )

    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--log-interval", help="Override log interval (default: -1, no change)", default=-1, type=int)
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). "
        "During hyperparameter optimization n-evaluations is used instead",
        default=25000,
        type=int,
    )
    parser.add_argument(
        "--optimization-log-path",
        help="Path to save the evaluation log and optimal policy for each hyperparameter tried during optimization. "
        "Disabled if no argument is passed.",
        type=str,
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument("--n-eval-envs", help="Number of environments for evaluation", default=1, type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=-1, type=int)
    parser.add_argument(
        "--save-replay-buffer", help="Save the replay buffer too (when applicable)", action="store_true", default=False
    )

    parser.add_argument(
        "-tags", "--wandb-tags", type=str, default=[], nargs="+", help="Tags for wandb run, e.g.: -tags optimized pr-123"
    )

    parser.add_argument(
        "-yaml",
        "--yaml-file",
        type=str,
        default=None,
        help="This parameter is deprecated, please use `--conf-file` instead",
    )

    parser.add_argument("-uuid", "--uuid", action="store_true", default=False, help="Ensure that the run has a unique ID")

    parser.add_argument("--wandb-entity", type=str, default="josh-holder", help="the entity (team) of wandb's project")

    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )

    return parser

def env_seed_uuid_setup(args):
    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env

    gym.envs.register(
            id='VisualMaze-v1',
            entry_point='custom_environments:VisualMazeEnv',
            max_episode_steps=100,
        )

    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr

    if args.yaml_file is not None:
        raise ValueError(
            "The`--yaml-file` parameter is deprecated and will be removed in RL Zoo3 v1.8, please use `--conf-file` instead",
        )

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()  # type: ignore[attr-defined]

    set_random_seed(args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    return args, env_id, uuid_str