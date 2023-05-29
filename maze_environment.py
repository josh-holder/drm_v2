import gym
import numpy as np
from gym.spaces import Discrete, Box
from typing import List, Optional

from contextlib import closing
from io import StringIO

from gymnasium.envs.toy_text.frozen_lake import generate_random_map, is_valid
from gym import Env, logger, spaces, utils

# Maze environment, heavily inspired by FrozenLake, but with images as the observation space,
# and accordingly the use of numbers rather than characters.

# 0 = path,
# 1 = wall.
# 2 = goal.
# 3 = start/player.

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# DFS to check that it's a valid path.
def is_valid(board: np.ndarray, max_size: int) -> bool:
    start_pos = np.where(board == 3)
    start_pos = (start_pos[0][0], start_pos[1][0])

    frontier, discovered = [], set()
    frontier.append(start_pos)
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
                if board[r_new,c_new] == 2:
                    return True
                if board[r_new,c_new] != 1:
                    frontier.append((r_new, c_new))
    return False

def generate_random_map(size: int = 8, p: float = 0.8) -> np.ndarray:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is a wall

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    while not valid:
        p = min(1, p)
        board = np.random.choice([0,1], (size, size), p=[p, 1 - p])

        #generate a random position for the start and the goal
        start_index = np.random.randint(0, size, 2)
        goal_index = np.random.randint(0, size, 2)

        #initialize start and goal positions
        board[start_index[0], start_index[1]] = 3
        board[goal_index[0], goal_index[1]] = 2
        valid = is_valid(board, size)
    return board

class MazeEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                size=8,
                prob_of_path=0.8,
                board=None,
                ):
        
        self.init_board = board #If the user provides a board, we always reset to the same board.

        self.size = size
        self.prob_of_path = prob_of_path

        self.nrow, self.ncol = nrow, ncol = size, size
        self.reward_range = (-1, 1)

        nA = 4
        nS = nrow * ncol

        self.action_space = Discrete(nA)

        self.observation_space = Box(low=0, high=3, shape=(size, size), dtype=np.uint8)

        self.render_mode = "human"

    def step(self, a):
        def update_board(board, new_row, new_col):
            board[board==3] = 0
            board[new_row, new_col] = 3

            return board

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, self.nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, self.ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)
    
        new_row, new_col = inc(self.s[0], self.s[1], a)

        new_position_value = self.board[new_row, new_col]

        if new_position_value == 1: # Hit a wall, so remain in the same state.
            return (self.board, -0.01, False, False, {"prob": 1})
        
        elif new_position_value == 0: #Moved onto a path, so update the state
            self.s = (new_row, new_col)
            self.board = update_board(self.board, new_row, new_col)
            return(self.board, -0.01, False, False, {"prob": 1})
        
        elif new_position_value == 2: #Reached goal, so end the episode.
            self.s = (new_row, new_col)
            self.board = update_board(self.board, new_row, new_col)
            return(self.board, 1, True, False, {"prob": 1})
        
        elif new_position_value == 3: #Failed to move (attempted to move into side of map), so stay in place.
            return(self.board, -0.01, False, False, {"prob": 1})
        
        else: 
            print("UNDEFINED STATE??")
            raise ValueError

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if self.init_board == None:
            self.board = generate_random_map(size=self.size, p=self.prob_of_path)
        else:
            self.board = self.init_board

        self.s = np.where(self.board == 3)
        self.s = (self.s[0][0], self.s[1][0])

        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return self.board, {"prob": 1}
    
    def render(self):
        board = self.board.tolist()
        outfile = StringIO()

        row, col = self.s[0], self.s[1]
        board = [[c for c in line] for line in board]
        board[row][col] = utils.colorize(board[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(str(line)) for line in board) + "\n")

        with closing(outfile):
            return outfile.getvalue()
    
if __name__ == "__main__":
    env = MazeEnv()

    current_obs, info = env.reset()

    done = False
    
    while not done:
        a = input()

        obs, r, done, _, info = env.step(int(a))
        print(obs)