import gym
import numpy as np
from gym.spaces import Discrete, Box
from typing import List, Optional

from gymnasium.envs.toy_text.frozen_lake import generate_random_map, is_valid

# Maze environment, heavily inspired by FrozenLake, but with images as the observation space,
# and accordingly the use of numbers rather than characters.

# 0 = frozen,
# 1 = hole.
# 2 = start.
# 3 = goal.

# DFS to check that it's a valid path.
def is_valid(board: np.ndarray, max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new,c_new] == "G":
                    return True
                if board[r_new,c_new] != "H":
                    frontier.append((r_new, c_new))
    return False

def generate_random_map(size: int = 8, p: float = 0.8) -> np.ndarray:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])

        #selec
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                size=8,
                p=0.8,
                desc=None,
                ):
        
        if desc is None: desc = generate_random_map(size=size,p=p)
        else: pass

        print(desc)

        self.action_space = Discrete(4)

        self.observation_space = Box(low=0, high=3, shape=(size, size), dtype=np.uint8)


    def step(self):
         pass
    
if __name__ == "__main__":
    env = MazeEnv()