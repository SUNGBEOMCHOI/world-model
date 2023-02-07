import copy
from collections import deque

import numpy as np
import gym
import imageio

from torchvision.transforms import transforms as T
from torchvision.transforms.functional import crop

class Env:
    def __init__(self):
        pass

    @staticmethod
    def make(env_name, env_cfg=None):
        if env_name == 'Breakout':
            return Breakout(env_cfg)

class Breakout(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self, env_cfg=None):
        self.env = gym.make('ALE/Breakout-v5')
        self.model = None
        self.total_reward = 0.0
        self.timestep = 0
        self.process = None
        self.is_save_video = False
        self.video_path = ''
        self.frames = []

    def step(self, action):
        self.timestep += 1   
        
        state, reward, done, info = self.env.step(action)
        self.frames.append(self.render())
        self.total_reward += reward

        state = self.preprocessing(state)

        if self.is_save_video and done:
            with imageio.get_writer(self.video_path, fps=30, macro_block_size = None) as video:
                for frame in self.frames:
                    video.append_data(frame)

        return state, reward, done, info
    
    def reset(self):
        self.total_reward = 0.0
        self.timestep = 0
        self.is_save_video = False
        self.video_path = ''
        self.frames = []

        state = self.env.reset()
        state = self.preprocessing(state)
        self.frames.append(self.render())
        return state

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.env.render(mode='rgb_array')
        elif mode == 'human':
            return self.env.render(mode='human')
        else:
            raise NotImplementedError

    def save_video(self, model=None, video_path='./video/test.mp4', ):
        self.is_save_video = True
        self.video_path = video_path
        self.model = model

    def preprocessing(self, state):
        if self.process == None:
            self.process = T.Compose([T.ToPILImage(),
                        T.Resize((64,64)),
                        T.ToTensor()]) # pixel value 0~255 -> 0~1

        state = self.process(state).cpu().detach().numpy()
        return state

if __name__ == "__main__":
    env = Env.make('Breakout')
    state = env.reset()
    done = False
    timestep = 0
    while not done:
        timestep += 1
        state, _, done, _ = env.step(0)
    print(timestep)
