import torch as th

def cliff_reward_shaping(state, action):
    shaped_states = th.zeros_like(state)

    top_row_states = [0,1,2,3,4,5,6,7,8,9,10]
    for top_row_state in top_row_states:
        push_right_rewards = th.where(th.logical_and(state==top_row_state,action==1), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_right_rewards)

    left_col_states = [12, 24, 36]
    for left_col_state in left_col_states:
        push_up_rewards = th.where(th.logical_and(state==left_col_state,action==0), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_up_rewards)

    right_col_states = [11, 23, 35]
    for right_col_state in right_col_states:
        push_down_rewards = th.where(th.logical_and(state==right_col_state,action==2), 1, 0)
        shaped_states = th.logical_or(shaped_states,push_down_rewards)

    rewards = th.where(shaped_states, 1, 0)

    # for i, reward in enumerate(list(rewards)):
    #     if reward == 0.75:
    #         print(f"r{reward}, state {state[i]}, act {action[i]}")

    return rewards