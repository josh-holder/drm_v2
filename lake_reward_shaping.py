import torch as th

def calc_lake_shaping_rewards(state, action):
    shaped_states = th.zeros_like(state)

    # push_right_states =  [20, 21, 42, 43, 24, 25, 26, 7, 8, 9, 10, 11, 12, 13, 14, 15, 96, 97, 98, 277, 278, 357, 358]
    # push_up_states =  [44, 27]
    # push_down_states =  [0, 22, 16, 36, 56, 76, 99, 119, 139, 159, 179, 199, 219, 237, 257, 279, 299, 317, 337, 359, 379]
    # push_left_states =  [239, 238, 319, 318]

    # push_right_states = [63, 64, 65, 66, 67, 68, 69] #default
    # push_right_states = [54, 55, 56, 57, 58, 59, 60] #small env
    push_right_states = [30, 31, 2, 3, 34, 35, 26, 27] #gate
    for push_right_state in push_right_states:
        push_right_rewards = th.where(th.logical_and(state==push_right_state,action==2), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_right_rewards)

    # push_up_states = [70,61,52,43]
    # push_up_states = [61, 52, 43]
    push_up_states = [32, 22, 12, 36]
    for push_up_state in push_up_states:
        push_up_rewards = th.where(th.logical_and(state==push_up_state,action==3), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_up_rewards)

    # push_down_states = [27, 36, 45, 54]
    # push_down_states = [27, 36, 45]
    push_down_states = [0, 10, 20, 4, 14, 24, 28]
    for push_down_state in push_down_states:
        push_down_rewards = th.where(th.logical_and(state==push_down_state,action==1), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_down_rewards)


    # for push_left_state in push_left_states:
    #     push_left_rewards = th.where(th.logical_and(state==push_left_state,action==0), 1, 0)
    #     shaped_states = th.logical_or(shaped_states,push_left_rewards)

    rewards = th.where(shaped_states, 0.015, 0)

    return rewards