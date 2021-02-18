from collections import Counter
from collections import deque
import random

import gym
from gym import error, spaces, utils
from gym.envs.classic_control import rendering
from gym.utils import seeding

import pyglet
from gym.envs.classic_control import rendering

class DrawText:
	def __init__(self, label:pyglet.text.Label):
		self.label=label
	def render(self):
		self.label.draw()

class Option(object):
	HUNGRY_FREQ=20
	ROW=7
	COL=11

class Action(object):
	RIGHT=0
	UP=1
	LEFT=2
	DOWN=3

class CellState(object):
	EMPTY = 0
	WALL = 1
	DOT = 2

class Reward(object):
	DOT = 1/3
	LOSE_STARVED = 0 #-1 #0 #-1/8
	LOSE_COLLISION = -1
	WON = 1
	IDLE = DOT/10/Option.HUNGRY_FREQ

dirs=[(0,1),(1,0),(0,-1),(-1,0)]

class SnakeGame(object):
	def __init__(self, head, desired_start_len):
		self.cur_step=0
		self.score=0
		self.snake = deque()
		self.empty_cells = [(y, x) for x in range(Option.COL) for y in range(Option.ROW)]
		self.dot = None
		self.prev_action=None

		self.push_head(head)
		for _ in range(desired_start_len-1):
			next_head=self.head()
			invalid_acts=set()
			while self.cell_state(next_head)!=CellState.EMPTY and len(invalid_acts)<2:
				invalid_acts.add(next_head)
				next_head=self.next_cell(random.randint(0,3))
			if len(invalid_acts)==2:
				break
			self.push_head(next_head)
			
		self.generate_dot()

	def cell_state(self, cell):
		if cell in self.empty_cells:
			return CellState.EMPTY
		if cell == self.dot:
			return CellState.DOT
		return CellState.WALL

	def head(self):
		return self.snake[0]
	def tail(self):
		return self.snake[-1]

	def push_head(self, cell):
		self.snake.appendleft(cell)
		if cell in self.empty_cells:
			self.empty_cells.remove(cell)
		if self.dot == cell:
			self.dot = None
	
	def push_tail(self, cell):
		self.snake.append(cell)
		if cell in self.empty_cells:
			self.empty_cells.remove(cell)
		if self.dot == cell:
			self.dot = None

	def pop_tail(self):
		self.empty_cells.append(self.snake.pop())
		return self.empty_cells[-1]

	def can_generate_dot(self):
		return len(self.empty_cells) > 0

	def generate_dot(self):
		self.dot = random.sample(self.empty_cells, 1)[0]
		self.empty_cells.remove(self.dot)
	
	def next_cell(self,action):
		dy,dx=dirs[action]
		hy,hx=self.head()
		hy=(hy+dy+Option.ROW)%Option.ROW
		hx=(hx+dx+Option.COL)%Option.COL
		return (hy,hx)
		

	def step(self, action):
		if self.prev_action==(action+2)%4:
			return Reward.LOSE_COLLISION
		self.prev_action=action
		
		next_head=self.next_cell(action)
		tcell=self.pop_tail()
		next_head_state = self.cell_state(next_head)
		self.push_tail(tcell)
		
		if next_head_state == CellState.WALL:    
			return Reward.LOSE_COLLISION
		self.push_head(next_head)
		if next_head_state == CellState.DOT:
			if self.can_generate_dot():
				self.generate_dot()
				return Reward.DOT                
			return Reward.WON

		self.cur_step+=1
		self.pop_tail()
		if self.cur_step%Option.HUNGRY_FREQ==0:
			self.pop_tail()
			if not self.snake:
				return Reward.LOSE_STARVED
		return len(self.snake)**1.5*Reward.IDLE


class SnakeEnv(gym.Env):
	metadata= {'render.modes': ['human']}

	def __init__(self):
		self.viewer = None

	def make_obs(self):
		obs = [[[0.0 for _ in range(Option.COL)] for __ in range(Option.ROW)] for ___ in range(2)]
		snake = self.game.snake
		
		for i in range(len(snake)):
			y,x=snake[i]
			obs[0][y][x]=(1-i/len(snake))/2
			if not i:
				obs[0][y][x]=1
		if len(snake)==1 and self.game.prev_action:
			ny,nx=self.game.next_cell((self.game.prev_action+2)%4)
			obs[0][ny][nx]=0.25
		#BUG??
		if not self.game.dot:
			self.game.generate_dot()
		obs[1][self.game.dot[0]][self.game.dot[1]]=1.0
		
		return (obs,self.game.cur_step%Option.HUNGRY_FREQ/Option.HUNGRY_FREQ)
	
	def step(self, action):
		reward = self.game.step(action)
		self.game.score+=reward
		done = reward in [Reward.LOSE_STARVED, Reward.LOSE_COLLISION, Reward.WON]
		info = None
	
		return self.make_obs(), reward, done, info

	def reset(self):
		self.start_pos = (random.randint(0,Option.ROW-1),random.randint(0,Option.COL-1))
		desired_start_len=random.randint(0,Option.ROW*Option.COL)
		self.game = SnakeGame(self.start_pos,desired_start_len)
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
				square.set_color(0,0,1)
			self.viewer.add_onetime(square)

		if self.game.dot:
			y, x = self.game.dot
			l, r, b, t = x*xr, (x+1)*xr, y*yr, (y+1)*yr
			square = rendering.FilledPolygon([(l,b), (r,b), (r,t), (l,t)])
			square.set_color(0, 1, 0)
			self.viewer.add_onetime(square)
			
		label = pyglet.text.Label("{:.2f}".format(self.game.score),
								  font_size=20, x=10, y=10,
								  anchor_x='left',anchor_y='bottom',
								  color=(255, 123, 255, 255))
		label.draw()
		self.viewer.add_onetime(DrawText(label))

		return self.viewer.render(return_rgb_array=mode=='rgb_array', )

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None

	def seed(self):
		pass
