import torch as th

def countbased_reward_scaling(count_dict, count_temp, states, actions):
    scaling = th.ones_like(states, dtype=th.float32)

    for i, (state, action) in enumerate(zip(states, actions)):
        if count_dict[(state.item(), action.item())] > 0:
            scaling[i] = 1/count_dict[(state.item(), action.item())]**count_temp

    return scaling