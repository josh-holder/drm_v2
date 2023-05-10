"""
Plot training reward/success rate
"""
import argparse
import os

import numpy as np
import seaborn
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func

# Activate seaborn
seaborn.set()

def plot_train():
    parser = argparse.ArgumentParser("Gather results, plot training reward/success")
    parser.add_argument("-a", "--algo", help="Algorithm to include", type=str, required=True)
    parser.add_argument("-e", "--env", help="Environment(s) to include", nargs="+", type=str, required=True)
    parser.add_argument("-f", "--exp-folder", help="Folders to include", type=str, required=True)
    parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
    parser.add_argument("--fontsize", help="Font size", type=int, default=14)
    parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
    parser.add_argument("-x", "--x-axis", help="X-axis", choices=["steps", "episodes", "time"], type=str, default="steps")
    parser.add_argument("-y", "--y-axis", help="Y-axis", choices=["success", "reward", "length"], type=str, default="reward")
    parser.add_argument("-w", "--episode-window", help="Rolling window size", type=int, default=100)
    parser.add_argument("--avg", "--plot-avg", help="Plot the average of all runs in this folder (0 for false, 1 for true)", type=int, default=0)

    args = parser.parse_args()

    algo = args.algo
    envs = args.env
    log_path = os.path.join(args.exp_folder, algo)

    x_axis = {
        "steps": X_TIMESTEPS,
        "episodes": X_EPISODES,
        "time": X_WALLTIME,
    }[args.x_axis]
    x_label = {
        "steps": "Timesteps",
        "episodes": "Episodes",
        "time": "Walltime (in hours)",
    }[args.x_axis]

    y_axis = {
        "success": "is_success",
        "reward": "r",
        "length": "l",
    }[args.y_axis]
    y_label = {
        "success": "Training Success Rate",
        "reward": "Training Episodic Reward",
        "length": "Training Episode Length",
    }[args.y_axis]

    dirs = []

    for env in envs:
        # Sort by last modification
        entries = sorted(os.scandir(log_path), key=lambda entry: entry.stat().st_mtime)
        dirs.extend(entry.path for entry in entries if env in entry.name and entry.is_dir())

    plt.figure(y_label, figsize=args.figsize)
    plt.title(y_label, fontsize=args.fontsize)
    plt.xlabel(f"{x_label}", fontsize=args.fontsize)
    plt.ylabel(y_label, fontsize=args.fontsize)
    for folder in dirs:
        try:
            data_frame = load_results(folder)
        except LoadMonitorResultsError:
            continue
        if args.max_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]
        try:
            y = np.array(data_frame[y_axis])
        except KeyError:
            print(f"No data available for {folder}")
            continue
        x, _ = ts2xy(data_frame, x_axis)

        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= args.episode_window:
            # Compute and plot rolling mean with window of size args.episode_window
            x, y_mean = window_func(x, y, args.episode_window, np.mean)
            plt.plot(x, y_mean, linewidth=2, label=folder.split("/")[-1])

    print(args.avg)
    if args.avg: 
        if args.x_axis != "steps": print("Not printing avg. because x axis is not steps.")
        else: plotAvgLine(dirs,args,x_axis,y_axis)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plotAvgLine(dirs,args,x_axis,y_axis):
    max_timesteps_of_any_run = 0

    all_results = []

    #grab all results
    for folder in dirs:
        try:
            data_frame = load_results(folder)
        except LoadMonitorResultsError:
            continue
        if args.max_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]

        all_results.append(data_frame)

        if max_timesteps_of_any_run < data_frame.l.cumsum().iloc[-1]:
            max_timesteps_of_any_run = data_frame.l.cumsum().iloc[-1]
        # try:
        #     y = np.array(data_frame[y_axis])
        # except KeyError:
        #     print(f"No data available for {folder}")
        #     continue
        # x, _ = ts2xy(data_frame, x_axis)

        # # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        # if x.shape[0] >= args.episode_window:
        #     # Compute and plot rolling mean with window of size args.episode_window
        #     x, y_mean = window_func(x, y, args.episode_window, np.mean)
        #     plt.plot(x, y_mean, linewidth=2, label=folder.split("/")[-1])
    
    mean_y_by_length = []
    lengths = np.linspace(0,max_timesteps_of_any_run, 1000)
    for length in lengths:
        ys = []
        for df in all_results:
            relevant_df = df[df.l.cumsum() <= length]
            if relevant_df.empty == False:
                ys.append(relevant_df[y_axis].iloc[-1])
        
        if len(ys) > 0:
            mean_y_by_length.append(sum(ys)/len(ys))
        else: mean_y_by_length.append(0)

    lengths, mean_y_rolling = window_func(np.array(lengths), np.array(mean_y_by_length), 5, np.mean)
    avg_data = np.hstack((lengths.reshape(-1, 1), mean_y_rolling.reshape(-1,1)))
    np.savetxt('avg_rollout.csv', avg_data, delimiter=',', header=f'{x_axis},{y_axis}')
    plt.plot(lengths, mean_y_rolling, linewidth=2, label="Average rewards")

if __name__ == "__main__":
    plot_train()
