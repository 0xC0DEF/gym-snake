from collections import Counter
from collections import deque
import random

import gym
from gym import error, spaces, utils
from gym.envs.classic_control import rendering
from gym.utils import seeding

class Option(object):
    HUNGRY_RATE=20
    ROW=8
    COL=8

class Action(object):
    LEFT=0
    FORWARD=1
    RIGHT=2

class CellState(object):
    EMPTY = 0
    WALL = 1
    DOT = 2

class Reward(object):
    ALIVE = -1/100
    DOT = 1
    DEAD = -2/3
    WON = 50

class SnakeGame(object):
    def __init__(self, head):
        self.cur_step=0

        self.snake = deque()
        self.empty_cells = {(x, y) for x in range(Option.COL) for y in range(Option.ROW)}
        self.dot = None
        self.dir=0

        self.add_to_head(head)
        self.generate_dot()

    def add_to_head(self, cell):
        self.snake.appendleft(cell)
        if cell in self.empty_cells:
            self.empty_cells.remove(cell)
        if self.dot == cell:
            self.dot = None

    def cell_state(self, cell):
        if cell in self.empty_cells:
            return CellState.EMPTY
        if cell == self.dot:
            return CellState.DOT
        return CellState.WALL

    def head(self):
        return self.snake[0]

    def remove_tail(self):
        tail = self.snake.pop()
        self.empty_cells.add(tail)

    def can_generate_dot(self):
        return len(self.empty_cells) > 0

    def generate_dot(self):
        self.dot = random.sample(self.empty_cells, 1)[0]
        self.empty_cells.remove(self.dot)
    
    def next_cell(self):
        dirs=[(0,1),(1,0),(0,-1),(-1,0)]
        dy,dx=dirs[self.dir]
        hy,hx=self.head()
        hy=(hy+dy+Option.ROW)%Option.ROW
        hx=(hx+dx+Option.COL)%Option.COL
        return (hy,hx)
        

    def step(self, action):
        if action==Action.LEFT:
            self.dir=(self.dir-1+4)%4
        if action==Action.RIGHT:
            self.dir=(self.dir+1)%4
        
        next_head=self.next_cell()
        next_head_state = self.cell_state(next_head)
        if next_head_state == CellState.WALL:
            return Reward.DEAD
        self.add_to_head(next_head)
        if next_head_state == CellState.DOT:
            if self.can_generate_dot():
                self.generate_dot()
                return Reward.DOT                
            return Reward.WON
        
        self.remove_tail()
        self.cur_step+=1
        if self.cur_step%Option.HUNGRY_RATE==0:
            self.remove_tail()
            if not self.snake:
                return Reward.DEAD
        return Reward.ALIVE


class SnakeEnv(gym.Env):
    metadata= {'render.modes': ['human']}

    def __init__(self):
        self.start_pos = (0, 0)
        self.game = SnakeGame(self.start_pos)
        self.viewer = None

    def make_obs(self):
        obs = [[[0.0 for _ in range(Option.COL)] for __ in range(Option.ROW)] for ___ in range(2)]
        for (y,x) in self.game.snake:
            obs[0][y][x]=1.0
        if self.game.snake:
            head=self.game.head()
            obs[0][head[0]][head[1]]=1.5
            ncell=self.game.next_cell()
            obs[0][ncell[0]][ncell[1]]=0.5
        
        obs[1][self.game.dot[0]][self.game.dot[1]]=1.0
        return obs
    
    def step(self, action):
        reward = self.game.step(action)
        done = reward in [Reward.DEAD, Reward.WON]
        info = None
    
        return self.make_obs(), reward, done, info

    def reset(self):
        self.game = SnakeGame(self.start_pos)
        return self.make_obs()

    def render(self, mode='human', close=False):
        width = height = 600
        width_scaling_factor = width / Option.COL
        height_scaling_factor = height / Option.ROW

        if self.viewer is None:
            self.viewer = rendering.Viewer(width, height)

        for x, y in self.game.snake:
            l, r, t, b = x*width_scaling_factor, (x+1)*width_scaling_factor, y*height_scaling_factor, (y+1)*height_scaling_factor
            square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            square.set_color(0, 0, 0)
            self.viewer.add_onetime(square)

        if self.game.dot:
            x, y = self.game.dot
            l, r, t, b = x*width_scaling_factor, (x+1)*width_scaling_factor, y*height_scaling_factor, (y+1)*height_scaling_factor
            square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            square.set_color(1, 0, 0)
            self.viewer.add_onetime(square)

        return self.viewer.render(return_rgb_array=mode=='rgb_array', )

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self):
        pass
