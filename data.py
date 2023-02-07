import os
import copy
import argparse
from collections import deque

import yaml
import numpy as np
import torch
from torch.utils.data import random_split, Dataset, DataLoader

from env import Env

def collect_dataset(cfg):
    """
    Collect train, test dataset for train V and M model
    """
    ########################
    #   Get configuration  #
    ########################
    env_name = cfg['env']['name']
    action_space = cfg['env']['action_space']
    data_path = cfg['data']['data_path']
    sample_size = cfg['data']['sample_size']

    replay = {'state':deque(maxlen=sample_size), 
                'action':deque(maxlen=sample_size),
                'reward':deque(maxlen=sample_size),
                'next_state':deque(maxlen=sample_size),
                'done':deque(maxlen=sample_size)}

    train_size = int(sample_size*0.9)
    test_size = sample_size - train_size

    os.makedirs(data_path, exist_ok=True)

    ########################
    #   Make Environment   #
    ########################
    env = Env.make(env_name)

    ########################
    #     Collect Data     #
    ########################
    while len(replay['state']) < sample_size:
        state = env.reset()
        done= False
        while not done:
            action = np.random.choice(action_space)
            next_state, reward, done, _ = env.step(action)

            replay['state'].append(copy.deepcopy(state))
            replay['action'].append(action)
            replay['reward'].append(reward)
            replay['next_state'].append(copy.deepcopy(next_state))
            replay['done'].append(done)

            state = next_state
    
    replay['state'] = np.array(replay['state'])
    replay['action'] = np.array(replay['action'])
    replay['reward'] = np.array(replay['reward'])
    replay['next_state'] = np.array(replay['next_state'])
    replay['done'] = np.array(replay['done'])

    np.savez(data_path+'/train',
            state=replay['state'][:train_size],
            action=replay['action'][:train_size],
            reward=replay['reward'][:train_size],
            next_state=replay['next_state'][:train_size],
            done=replay['done'][:train_size]
            )

    np.savez(data_path+'/test',
            state=replay['state'][train_size:],
            action=replay['action'][train_size:],
            reward=replay['reward'][train_size:],
            next_state=replay['next_state'][train_size:],
            done=replay['done'][train_size:]
            )

    print(f"------ {train_size} train, {test_size} test data samples collected ------")

class EnvDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

        self.state = self.data['state']
        self.action = self.data['action']
        self.reward = self.data['reward']
        self.next_state = self.data['next_state']
        self.done = self.data['done']

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        state = torch.tensor(self.state[idx], dtype=torch.float32, requires_grad=False)
        action = torch.tensor(self.action[idx], dtype=torch.int8, requires_grad=False)
        reward = torch.tensor(self.reward[idx], dtype=torch.float32, requires_grad=False)
        next_state = torch.tensor(self.next_state[idx], dtype=torch.float32, requires_grad=False)
        done = torch.tensor(self.done[idx], dtype=torch.bool, requires_grad=False)
        return state, action, reward, next_state, done

def get_train_dataset(data_path='./data'):
    """
    Return dataset for training and validation

    Args:

    Returns:
        train_dataset
        validation_dataset
    """
    train_dataset = EnvDataset(data_path+'./train.npz')

    train_size = int(len(train_dataset) * 0.9)
    validation_size = int(len(train_dataset) * 0.1)

    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

    return train_dataset, validation_dataset
    

def get_test_dataset(data_path='./data'):
    """
    Return dataset for test

    Args:

    Returns:
        test_dataset
    """
    test_dataset = EnvDataset(data_path+'./train.npz')
    return test_dataset

def get_dataloader(dataset, batch_size, train=True):
    """
    Return torch dataloader for training/validation

    Args:
        dataset: List of torch dataset
        batch_size

    Returns:
        data_loader: List of torch data loader
    """
    if train:
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/breakout_config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    collect_dataset(cfg)