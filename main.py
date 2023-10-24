import os
import sys
import torch
import time
from time import strftime, localtime, time
import numpy as np
import random
import gym
from tensorboardX import SummaryWriter
from pprint import pprint

from models.graph_attention import DRL_GAT
from utils import *
from envs import make_vec_envs
from ppo import PPO_Training
from utils import get_args, registration_envs, load_ppo_policy, Logger

def main(args):
    box_setting = 'continuous' if args.continuous else 'discrete'
    timeStr = args.training_algorithm + '_setting{}_{}_'.format(args.setting, box_setting) + \
              strftime('%Y.%m.%d-%H-%M-%S', localtime(time())) + '_{}'.format(round(np.random.rand(), 5))

    os.makedirs('{}/logger/train'.format(args.log_path), exist_ok=True)
    sys.stdout = Logger('{}/logger/train/{}.log'.format(args.log_path, timeStr), sys.stdout)
    sys.stderr = Logger('{}/logger/train/{}.log'.format(args.log_path, timeStr), sys.stderr)
    pprint(vars(args))

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)

    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    writer_path = '{}/summary_writer/{}'.format(args.log_path, timeStr)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    writer = SummaryWriter(logdir=writer_path)
    envs = make_vec_envs(args, None, True)

    bppObs_size = (args.num_box + args.num_next_box + args.num_candidate_action) * args.node_dim
    BPP_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(bppObs_size, )),
                         gym.spaces.Discrete(n=args.num_candidate_action),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         args.num_candidate_action,
                         args.node_dim,
                         policy_ctg='place',
                         )

    advObs_size = (args.num_box + args.num_next_box) * args.node_dim
    adv_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(advObs_size, )),
                         gym.spaces.Discrete(n=args.num_next_box),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         0,
                         args.node_dim,
                         policy_ctg='permutation',
                         )

    mixObs_size = (args.num_box + args.num_next_box) * args.node_dim
    mix_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(mixObs_size,)),
                         gym.spaces.Discrete(n=args.num_next_box),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         0,
                         args.node_dim,
                         policy_ctg='permutation',
                         )

    BPP_policy = BPP_policy.to(device)
    adv_policy = adv_policy.to(device)
    mix_policy = mix_policy.to(device)

    if args.load_bpp_model:
        BPP_policy = load_ppo_policy(args.bpp_model_path, BPP_policy)
    if args.load_adv_model:
        adv_policy = load_ppo_policy(args.adv_model_path, adv_policy)
    if args.load_mix_model:
        mix_policy = load_ppo_policy(args.mix_model_path, mix_policy)


    train_model = PPO_Training(writer,
                               timeStr,
                               BPP_policy,
                               adv_policy,
                               mix_policy,
                               args,)
    train_model.train_n_steps(envs, args, device)


if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)


