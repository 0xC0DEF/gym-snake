from collections import Counter
from collections import deque
import random

import gym
from gym import error, spaces, utils
from gym.envs.classic_control import rendering
from gym.utils import seeding

class Option(object):
    HUNGRY_RATE=20
    ROW=7
    COL=11

class Action(object):
    LEFT=0
    FORWARD=1
    RIGHT=2

class CellState(object):
    EMPTY = 0
    WALL = 1
    DOT = 2

class Reward(object):
    ALIVE = -1/20
    DOT = 1
    DEAD = -2
    WON = 50

class SnakeGame(object):
    def __init__(self, head):
        self.cur_step=0

        self.snake = deque()
        self.empty_cells = {(y, x) for x in range(Option.COL) for y in range(Option.ROW)}
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
            self.dir=(self.dir+1+4)%4
        if action==Action.RIGHT:
            self.dir=(self.dir-1+4)%4
        
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
        obs = [[[0.0 for _ in range(Option.COL)] for __ in range(Option.ROW)] for ___ in range(3)]
        snake = self.game.snake
        
        for i in range(len(snake)):
            y,x=snake[i]
            obs[0][y][x]=2-i/len(snake)
        
        if self.game.snake:
            hy,hx=self.game.head()
            obs[2][hy][hx]=1
            ny,nx=self.game.next_cell()
            obs[2][ny][nx]=2
        
        obs[2][self.game.dot[0]][self.game.dot[1]]=1.0
        
        return obs
    
    def step(self, action):
        reward = self.game.step(action)
        done = reward in [Reward.DEAD, Reward.WON]
        info = None
    
        return self.make_obs(), reward, done, info

    def reset(self):
        self.game = SnakeGame(self.start_pos)
        return self.make_obs()

    def render(self, mode='human'):
        width=0
        height=0
        if Option.COL>Option.ROW:
            width=600
            height=int(600*Option.ROW/Option.COL)
        else:
            width=int(600*Option.ROW/Option.COL)
            height=600
        xr = width / Option.COL
        yr = height / Option.ROW

        if self.viewer is None:
            self.viewer = rendering.Viewer(width,height)
            
        bg = rendering.FilledPolygon([(0,0), (width,0), (width,height), (0,height)])
        bg.set_color(0,0,0)
        self.viewer.add_onetime(bg)

        for i in range(len(self.game.snake)):
            y,x=self.game.snake[i]
            l, r, b, t = x*xr, (x+1)*xr, y*yr, (y+1)*yr
            square = rendering.FilledPolygon([(l,b), (r,b), (r,t), (l,t)])
            c=1-i/len(self.game.snake)*3/4
            square.set_color(c,c,c)
            if not i:
                square.set_color(1,0,0)
            self.viewer.add_onetime(square)

        if self.game.dot:
            y, x = self.game.dot
            l, r, b, t = x*xr, (x+1)*xr, y*yr, (y+1)*yr
            square = rendering.FilledPolygon([(l,b), (r,b), (r,t), (l,t)])
            square.set_color(0, 1, 0)
            self.viewer.add_onetime(square)

        return self.viewer.render(return_rgb_array=mode=='rgb_array', )

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self):
        pass
