import torch as th

def lake_reward_shaping(state, action):
    shaped_states = th.zeros_like(state)

    push_right_states =  [17, 18, 45, 46, 21, 22, 16, 29, 31, 42, 44]
    push_up_states =  [69, 56, 43, 30, 47, 34, 31, 35, 58, 60]
    push_down_states =  [19, 32, 23, 4, 5, 6, 8, 9, 10]
    push_left_states =  [24, 31, 35, 44, 48, 57]

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