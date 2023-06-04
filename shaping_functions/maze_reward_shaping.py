import torch as th

PATH = 0
WALL = 85
GOAL = 170
PLAYER = 255

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

def maze_reward_shaping(state, action):
    """
    Reward shaping based on Manhattan distance.

    Input: (batch_size x 1 x maze_height x maze_width)
    """
    action = th.squeeze(action) #turn from (batch_size,1) -> (batch_size,)

    #Find current x,y positions (batch_size x 2)
    #Get 2 (batch_size,) tensors which determine the row and column of the player/goal location 
    #in each state from the batch. Then, stack them together to get a (batch_size,2) tensor.
    player_locations = th.vstack(th.where(state==PLAYER)[2:])
    player_locations = th.transpose(player_locations, 0, 1)

    goal_locations = th.vstack(th.where(state==GOAL)[2:])
    goal_locations = th.transpose(goal_locations, 0, 1)

    go_down = th.where(goal_locations[:,0]>player_locations[:,0],1,0) #1 if you should go right, 0 if you shouldn't
    go_up = th.where(goal_locations[:,0]<player_locations[:,0],1,0)
    go_left = th.where(goal_locations[:,1]<player_locations[:,1],1,0)
    go_right = th.where(goal_locations[:,1]>player_locations[:,1],1,0)

    right_rewards = th.where(action==RIGHT, go_right, 0)
    left_rewards = th.where(action==LEFT, go_left, 0)
    up_rewards = th.where(action==UP, go_up, 0)
    down_rewards = th.where(action==DOWN, go_down, 0)

    rewards = right_rewards + left_rewards + up_rewards + down_rewards

    for i, reward in enumerate(rewards[0:10]):
        print(state[i,:,:,:])
        print(f"Action: {action[i]}")
        print(f"Reward: {reward}")
        
    reward_scale = 0.005 #negative step reward is -0.01, so this is 1.5x that

    return reward_scale*rewards.unsqueeze(1) #return a (batch_size,1) tensor instead of a (batch_size,) tensor