# Custom reward function to use in reward shaping
# Calculates a "reward" value given a state and action
# based on the action and the heuristic action compare

# PARAMETERS
# env: the environment
# state: current state of the environment and agent
# action: calculated next action based on the <TODO> algorithm

# RETURNS:
# Reward value that represents how different the given action
# and the action calculated by the heuristic are

# Currently this value will be non-positive, with 0 being
# that the two actions are exactly the same, and gets more
# negative the more different they are.
import numpy as np
import torch as th

def lander_reward_shaping(state, action):
	"""
	calculate the action the heuristic would choose, based on https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
	Then, compare that to the chosen action and assign a shaping reward to the agent

	Input:
	 - state: (batch size)x(8) tensor with current observations
	 - action: (batch_size)x(2) tensor with current action

	Output:
	 - rewards: (batch_size)x(1) tensor with shaping rewards for each (s,a) pair in the batch.
	"""
	#Calculate angle target, clipped between +- 0.4
	angle_targ = th.mul(state[:,0],0.5) + state[:,2]
	angle_targ = th.clip(angle_targ, -0.4, 0.4)

	hover_targ = th.mul(th.abs(state[:,0]), 0.55)

	angle_todo = th.mul((angle_targ - state[:,4]),0.5) - state[:,5]
	hover_todo = th.mul((hover_targ - state[:,1]),0.5) - th.mul(state[:,3],0.5)

	#If the lander is touching the ground, adjust the targets
	landing_leg_down = (state[:,6]+state[:,7]).bool()
	angle_todo = th.where(landing_leg_down,0,angle_todo)
	hover_todo = th.where(landing_leg_down,th.mul(state[:,3],-0.5), hover_todo)

	heuristic_action = th.cat((th.sub(th.mul(hover_todo,20),1), th.mul(angle_todo,-20)))
	heuristic_action = th.clip(heuristic_action, -1, +1)

	# calculate the difference between the algorithm action and the heuristic action
	# currently this is just a naive method, where the reward is the total negative
	# absolute value difference between the two actions.
	# We might look into more sophisticated methods if time allows
	rewards = th.mul((th.abs(heuristic_action[0] - action[:,0]) + th.abs(heuristic_action[1] - action[:,1])), -1)

	return rewards.unsqueeze(1) #convert from 100 -> 100x1 before returning
