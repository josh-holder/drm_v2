import torch as th

def car_reward_shaping(state, action):
    # If the velocity is positive, reward positive actions according to action effort
    # If the velocity is negative, incentivize a small negative action.
    # This will cause it to take many attempts for the car to get up the hill.
    rewards = th.where(th.gt(state[:,1],0).unsqueeze(1),action,(-abs(action+0.1)+0.1)*10)
    return rewards