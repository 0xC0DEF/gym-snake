# gym-snake

1) Clone the repo:
```
$ git clone git@github.com:SeanBae/gym-snake.git
```

2) `cd` into `gym-snake` and run:
```
$ pip install -e .
```

3) Test it
```
import gym
import gym_snake
import time
env = gym.make('snake-v0')

obs=env.reset()
#0: left turn
#1: go ahead
#2: right turn
acts=[1,1,1,0,1,1,2,1,1]
for i in acts:
    obs, rwd, done, _ = env.step(i)
    env.render()
    time.sleep(1)
env.close()
```
