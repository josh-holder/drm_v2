import torch as th

def car_reward_shaping(state, action):
    # If the velocity is positive, reward positive actions according to action effort
    # If the velocity is negative, incentivize a small negative action.
    # This will cause it to take many attempts for the car to get up the hill.
    rewards = th.where(th.gt(state[:,1],0).unsqueeze(1),action,(-abs(action+0.1)+0.1)*10)
    return rewards

def will_make_it_up_hill_with_neg_action(state):
    force = -0.5
    power = 0.0015
    cos_const = -0.0025

    accel = force*power + cos_const

    time_to_zero = th.div(state[:,1],-accel)
    time_to_zero = th.clip(time_to_zero, 0, 1000).unsqueeze(1) #clip to avoid negative times

    xf = state[:,0].unsqueeze(1) + th.mul(state[:,1].unsqueeze(1),time_to_zero) + 0.5*accel*th.pow(time_to_zero,2)

    # print(th.hstack((xf, time_to_zero, xf+time_to_zero**2)))

    successes = th.where(xf >= 0.45,1,0).bool() #unsqueeze to (batch_size)x1 from (batchsize)

    return successes