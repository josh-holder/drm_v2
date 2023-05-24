import torch as th

def lake_reward_shaping(state, action):
    shaped_states = th.zeros_like(state)

    push_right_states =  [62, 29, 30]
    push_up_states =  [63, 46, 47, 80]
    push_down_states =  [31, 12, 13, 14]
    push_left_states =  [32, 47, 64]

    for push_right_state in push_right_states:
        push_right_rewards = th.where(th.logical_and(state==push_right_state,action==2), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_right_rewards)

    for push_up_state in push_up_states:
        push_up_rewards = th.where(th.logical_and(state==push_up_state,action==3), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_up_rewards)

    for push_down_state in push_down_states:
        push_down_rewards = th.where(th.logical_and(state==push_down_state,action==1), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_down_rewards)

    for push_left_state in push_left_states:
        push_left_rewards = th.where(th.logical_and(state==push_left_state,action==0), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_left_rewards)

    rewards = th.where(shaped_states, 0.015, 0)

    return rewards