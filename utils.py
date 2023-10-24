import os, sys
from shutil import copyfile, copytree
import argparse
import numpy as np
from gym.envs.registration import register
import gym
from time import time
import re
import random
import glob

import torch
import torch.nn as nn

import matplotlib
if sys.platform != 'linux': matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='3D Bin Packing Arguments')
    parser.add_argument('--setting', type=int, default=1,
                        help='Experiemnt setting: 1 | 2 | 3')
    parser.add_argument('--container-size', type=float, default=10,
                        help='The width, length and height of the container')
    parser.add_argument('--max-item-size', type=int, default=5,
                        help='the maximum size of box')
    parser.add_argument('--min-item-size', type=int, default=1,
                        help='the minimum size of box')
    parser.add_argument('--continuous', action='store_true', default=False,
                        help='Use continuous environemnt or discrete envvironment')
    parser.add_argument('--num-box', type=int, default=80,
                        help='The maximum number of nodes to represent the bin state')
    parser.add_argument('--num-next-box', type=int, default=5,
                        help='The maximum number of next box')
    parser.add_argument('--num-candidate-action', type=int, default=120,
                        help='The maximum number of particles to represent the feasible actions')
    parser.add_argument('--node-dim', type=int, default=9,
                        help='The vector size to represent one node')
    parser.add_argument('--sparse-reward', type=int, default=1,
                        help='The reward from env can be dense (0) or sparse (1)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Donot use cuda')
    parser.add_argument('--device', type=int, default=0,
                        help='Choose GPUs to train model')
    parser.add_argument('--seed', type=int, default=0,
                        help='Set the random seed')

    parser.add_argument('--training-algorithm', type=str, default='ppo',
                        help='Choose one training algorithm: ppo')
    parser.add_argument('--num-processes', type=int, default=64,
                        help='The number of parallel processes used for training')
    parser.add_argument('--num-steps', type=int, default=30,
                        help='The rollout length for n-step training')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='The maximum norm of gradients')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor of Return')

    parser.add_argument('--embedding-size', type=int, default=64,
                        help='Dimension of the input embedding')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Dimension of hidden layers')
    parser.add_argument('--gat-layer-num', type=int, default=1,
                        help='The number of GAT layers')

    parser.add_argument('--model-save-interval', type=int, default=200,
                        help='The model saving frequency')
    parser.add_argument('--model-update-interval', type=int, default=20e30,
                        help='The frequency of creaing new model')
    parser.add_argument('--model-save-path', type=str, default='experiment',
                        help='The path to save the trained BPP model')
    parser.add_argument('--print-log-interval', type=int, default=10,
                        help='The frequency of printing the training logs')
    parser.add_argument('--max-model-num', type=int, default=5)
    parser.add_argument('--log-path', type=str, default='./logs')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Evaluation mode')
    parser.add_argument('--evaluation-episodes', type=int, default=100,
                        help='The number of evaluated episodes')
    parser.add_argument('--load-bpp-model', action='store_true', default=False,
                        help='Load the trained BPP model')
    parser.add_argument('--load-adv-model', action='store_true', default=False,
                        help='Load the trained Adv model')
    parser.add_argument('--load-mix-model', action='store_true', default=False,
                        help='Load the trained mixture model')
    parser.add_argument('--bpp-model-path', type=str,
                        help='The path to load BPP model')
    parser.add_argument('--adv-model-path', type=str,
                        help='The path to load Adv model')
    parser.add_argument('--mix-model-path', type=str,
                        help='The path to load Mix model')
    parser.add_argument('--load-dataset', action='store_true', default=False,
                        help='Load an existing dataset')
    parser.add_argument('--dataset-path', type=str,
                        help='The path to load dataset')

    parser.add_argument('--sample-from-distribution', action='store_true', default=False,
                        help='Sample continuous item size from a Uniform distribution')
    parser.add_argument('--sample-left-bound', type=float, default=1.,
                        help='The left bound of the uniform distribution')
    parser.add_argument('--sample-right-bound', type=float, default=5.,
                        help='the right bound of the uniform distribution')
    parser.add_argument('--unit-interval', type=float, default=1.,
                        help='the unit interval for height samples')

    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Set learning rate for a2c (default: 1e-6) or ppo (default: 3e-4)')
    parser.add_argument('--minimum-lr', type=float, default=1e-4,
                        help='the minimum learning rate for ppo (default: 1e-4)')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='choose whether to linearly decay to learning rate or not')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--begin-decay-step', type=int, default=12000,
                        help='decay the learning rate from this step')

    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='choose whether to use GAE for advantage approximation or not')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--clip-param', type=float, default=0.1,
                        help='ppo clip parameter (default: 0.1)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--ppo-epoch', type=int, default=1,
                        help='number of ppo epochs (default: 10)')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False,
                        help='compute returns taking into account time limits')
    parser.add_argument('--value-loss-coef', type=float, default=1.,
                        help='The coefficient of value loss of PPO')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='The coefficient of entropy of PPO')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')

    parser.add_argument('--bpp-update-steps', type=int, default=10)
    parser.add_argument('--adv-update-steps', type=int, default=10)
    parser.add_argument('--mix-update-steps', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)

    args = parser.parse_args()
    args.no_cuda = args.no_cuda | (not torch.cuda.is_available())

    if args.no_cuda:
        args.device = 'cpu'

    args.container_size = int(args.container_size) if not args.continuous else args.container_size
    args.container_size = 3 * (args.container_size,)
    args.env_id = '3dBP-Discrete-v0' if not args.continuous else '3dBP-Continuous-v0'
    args.item_size_set = DiscreteBoxData(lower=args.min_item_size, higher=args.max_item_size)
    args.normFactor = 1. / max(args.container_size)
    args.num_processes = 1 if args.evaluate else args.num_processes
    args.model_save_path = os.path.join(args.log_path, args.model_save_path)

    return args


def DecodeObs4Place(observation, bin_node_len, box_node_len, candidata_node_len, node_dim):
    bin_node = observation[:, 0:bin_node_len, 0:node_dim-2]
    box_node = observation[:, bin_node_len:bin_node_len+box_node_len, 3:node_dim-2]
    candidata_node = observation[:, bin_node_len+box_node_len:, 0:node_dim-1]

    assert candidata_node.size(1) == candidata_node_len
    assert observation[:, bin_node_len:bin_node_len+box_node_len, 0:3].sum() == 0

    full_mask = observation[:, :, -1]
    valid_mask = observation[:, bin_node_len+box_node_len:, node_dim-2]

    return bin_node, box_node, candidata_node, valid_mask, full_mask

def DecodeObs4Adv(observation, bin_node_len, box_node_len, node_dim):
    bin_node = observation[:, 0:bin_node_len, 0:node_dim-2]
    box_node = observation[:, bin_node_len:bin_node_len+box_node_len, 3:node_dim-2]

    assert observation[:, bin_node_len:bin_node_len+box_node_len, 0:3].sum() == 0

    full_mask = observation[:, :, -1]
    valid_mask = observation[:, bin_node_len:, -1]

    return bin_node, box_node, valid_mask, full_mask

def DecodeObs4Critic(observation, bin_node_len, box_node_len, node_dim):
    bin_node = observation[:, 0:bin_node_len, 0:node_dim-2]
    box_node = observation[:, bin_node_len:bin_node_len+box_node_len, 3:node_dim-2]

    assert observation[:, bin_node_len:bin_node_len+box_node_len, 0:3].sum() == 0

    full_mask = observation[:, 0:bin_node_len+box_node_len, -1]
    return bin_node, box_node, full_mask


class Logger(object):
    def __init__(self, file_name='logging.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, 'ab', buffering=0)
        self.log_disable = False
    def write(self, message):
        self.terminal.write(str(message))
        if not self.log_disable:
            self.log.write(str(message).encode('utf-8'))
    def flush(self):
        self.terminal.flush()
        if not self.log_disable:
            self.log.flush()
    def close(self):
        self.log.close()
    def disable_log(self):
        self.log_disable = True
        self.log.close()


def load_ppo_policy(model_path, policy_model):
    assert os.path.exists(model_path), 'File does not exist'

    if sys.platform is not 'linux':
        pretrained_state_dict = torch.load(model_path, map_location='cpu')
    else:
        pretrained_state_dict = torch.load(model_path)

    policy_model.load_state_dict(pretrained_state_dict, strict=True)
    #print('Load trained model', model_path)
    return policy_model


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, minimum_lr=1e-4):
    """Decreases the learning rate linearly"""
    if epoch - 12000 >= 0:
        decay_ratio = epoch / float(total_num_epochs) if epoch < total_num_epochs else \
            float(total_num_epochs-1)/float(total_num_epochs)
        lr = initial_lr - (initial_lr * decay_ratio)
        lr = minimum_lr if lr < minimum_lr else lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def DiscreteBoxData(lower=1, higher=5, resolution=1):
    item_size_set = []
    for i in range(lower, higher + 1):
        for j in range(lower, higher + 1):
            for k in range(lower, higher + 1):
                item_size_set.append((i * resolution, j * resolution, k * resolution))
    return item_size_set

def registration_envs():
    register(
        id='3dBP-Discrete-v0',
        entry_point='3dBP_envs.3dBP_Discrete0:PackingDiscrete',
    )
    register(
        id='3dBP-Continuous-v0',
        entry_point='3dBP_envs.3dBP_Continuous0:PackingContinuous',
    )



def generate_discrete_dataset(dataset_size=100, traj_len=150):
    box_set = DiscreteBoxData()
    box_density = lambda: -np.random.random((traj_len, 1)) + 1

    dataset = []
    for i in range(dataset_size):
        idx = np.random.choice(np.arange(len(box_set)), size=traj_len, replace=True)
        boxes = np.concatenate((np.array(box_set)[idx, ...], box_density()), axis=-1)
        dataset.append(boxes)

    torch.save(dataset, './datasets/discrete_dataset.pt')
    return


def generate_continuous_dataset(dataset_size=100, traj_len=150,
                                sample_left_bound=0.1, sample_right_bound=0.5, unit_interval=0.1):
    unit_num = int((sample_right_bound - sample_left_bound + unit_interval) / unit_interval)
    GenNextBox = lambda: [
        round(np.random.uniform(sample_left_bound, sample_right_bound), 3),
        round(np.random.uniform(sample_left_bound, sample_right_bound), 3),
        np.random.choice(np.linspace(start=sample_left_bound, stop=sample_right_bound, num=unit_num))
    ]
    box_density = lambda: -np.random.random((traj_len, 1)) + 1

    dataset = []
    for i in range(dataset_size):
        boxes = np.concatenate(([GenNextBox() for _ in range(traj_len)], box_density()), axis=-1)
        dataset.append(boxes)

    torch.save(dataset, './datasets/continuous_dataset.pt')
    return


if __name__ == '__main__':
    pass