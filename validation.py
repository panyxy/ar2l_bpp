import os, sys
import time
from models.graph_attention import DRL_GAT
from utils import registration_envs, get_args, load_ppo_policy, Logger
import gym
import torch
from time import strftime, localtime, time
import glob

import copy
import numpy as np
import utils
from time import clock
import pandas as pd
from pprint import pprint


def main(args):
    box_setting = 'continuous' if args.continuous else 'discrete'
    timeStr = args.training_algorithm + '_setting{}_{}_'.format(args.setting, box_setting) + \
              strftime('%Y.%m.%d-%H-%M-%S', localtime(time())) + '_{}'.format(round(np.random.rand(), 5))

    os.makedirs('{}/logger/val'.format(args.log_path), exist_ok=True)
    sys.stdout = Logger('{}/logger/val/{}.log'.format(args.log_path, timeStr), sys.stdout)
    sys.stderr = Logger('{}/logger/val/{}.log'.format(args.log_path, timeStr), sys.stderr)
    pprint(vars(args))

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists('{}/validation/{}'.format(args.log_path, timeStr)):
        os.makedirs('{}/validation/{}'.format(args.log_path, timeStr))

    # Create single packing environment and load existing dataset.
    advObs_size = (args.num_box + args.num_next_box) * args.node_dim
    adv_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(advObs_size,)),
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
    adv_policy = adv_policy.to(device)

    bppObs_size = (args.num_box + args.num_next_box + args.num_candidate_action) * args.node_dim
    BPP_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(bppObs_size,)),
                         gym.spaces.Discrete(n=args.num_candidate_action),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         args.num_candidate_action,
                         args.node_dim,
                         policy_ctg='place'
                         )
    BPP_policy = BPP_policy.to(device)

    adv_files = sorted(glob.glob(os.path.join(args.adv_model_path, 'Adv*.pt')))
    bpp_files = sorted(glob.glob(os.path.join(args.bpp_model_path, 'BPP*.pt')))
    n_model = min(len(adv_files), len(bpp_files))

    args.adv_model_index = np.arange(0, n_model)
    args.bpp_model_index = np.arange(0, n_model)

    saved_ratio = dict()
    saved_length = dict()
    for adv_idx, bpp_idx in zip(args.adv_model_index, args.bpp_model_index):
        adv_idx, bpp_idx = int(adv_idx), int(bpp_idx)

        envs = gym.make(args.env_id,
                        setting=args.setting,
                        container_size=args.container_size,
                        item_set=args.item_size_set,
                        data_name=args.dataset_path,
                        load_test_data=args.load_dataset,
                        num_box=args.num_box,
                        num_next_box=args.num_next_box,
                        candidate_pos_nums=args.num_candidate_action,
                        node_dim=args.node_dim,
                        sample_from_distribution=args.sample_from_distribution,
                        sample_left_bound=args.sample_left_bound,
                        sample_right_bound=args.sample_right_bound,
                        unit_interval=args.unit_interval,
                        sparse_reward=args.sparse_reward,
                        )

        adv_policy = load_ppo_policy(adv_files[adv_idx], adv_policy)
        BPP_policy = load_ppo_policy(bpp_files[bpp_idx], BPP_policy)

        episode_ratio, episode_length = evaluate_one_dist(
            BPP_policy, adv_policy, envs, timeStr, args, device,
            eval_freq=args.evaluation_episodes, factor=args.normFactor
        )

        saved_ratio['{}_{}'.format(adv_idx, bpp_idx)] = episode_ratio
        saved_length['{}_{}'.format(adv_idx, bpp_idx)] = episode_length

        print('Model: {}, Ratio Mean: {:.5f}, Ratio Std: {:.5f}, Num: {:.5f}'.format(
            (adv_idx, bpp_idx), np.mean(episode_ratio), np.std(episode_ratio), np.mean(episode_length)
        ))

        saved_ratio = pd.DataFrame(saved_ratio)
        saved_length = pd.DataFrame(saved_length)

        saved_ratio.to_csv(os.path.join('{}/validation'.format(args.log_path), timeStr, 'ratio.csv'))
        saved_length.to_csv(os.path.join('{}/validation'.format(args.log_path), timeStr, 'length.csv'))


def evaluate_one_dist(BPP_policy, adv_policy, eval_envs, timeStr, args, device, eval_freq=100, factor=1):

    num_box, num_next_box, num_candidate_action = \
        args.num_box, args.num_next_box, args.num_candidate_action
    node_dim = args.node_dim

    BPP_policy.eval()
    adv_policy.eval()

    adv_obs = eval_envs.reset()
    adv_obs = adv_obs.reshape(1, num_box + num_next_box, args.node_dim)
    adv_obs = torch.FloatTensor(adv_obs).to(device)
    bpp_obs, box_idx = execute_permute_policy(adv_policy, eval_envs, adv_obs, device, args)

    step_counter = 0
    episode_ratio = []
    episode_length = []
    all_episodes = []

    while step_counter < eval_freq:
        with torch.no_grad():
            action_log_probs, action, entropy = BPP_policy.forward_actor(bpp_obs, deterministic=True, normFactor=factor)

        location = bpp_obs.split([num_box + num_next_box, num_candidate_action], dim=1)[-1][[0], action.squeeze(1)][:, :7]
        zero_padding = torch.zeros((location.size(0), 1)).to(device)
        execution = torch.cat((location, box_idx, zero_padding), dim=-1)
        adv_obs, reward, done, infos = eval_envs.step(execution[0].cpu().numpy())

        if done:
            if 'ratio' in infos.keys():
                episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                episode_length.append(infos['counter'])

            items = [
                (item.x, item.y, item.z, item.w, item.h, item.l, item.density) for item in eval_envs.space.boxes[1:]
            ]
            all_episodes.append(items)
            step_counter += 1
            adv_obs = eval_envs.reset()

        adv_obs = adv_obs.reshape(1, num_box + num_next_box, args.node_dim)
        adv_obs = torch.FloatTensor(adv_obs).to(device)
        bpp_obs, box_idx = execute_permute_policy(adv_policy, eval_envs, adv_obs, device, args)

    return episode_ratio, episode_length


def execute_permute_policy(adv_policy, envs, adv_obs, device, args):
    num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
    node_dim = args.node_dim

    if args.load_adv_model:
        with torch.no_grad():
            _, box_idx, _ = adv_policy.forward_actor(adv_obs, deterministic=True, normFactor=args.normFactor)
    else:
        box_idx = torch.zeros(adv_obs.size(0))[:, None].to(device).to(torch.long)

    tmp_box = adv_obs.split([num_box, num_next_box, ], dim=1)[1][[0], box_idx.squeeze(1)][:, :7]
    one_padding = torch.ones((tmp_box.size(0), 1)).to(device)
    adv_act = torch.cat((tmp_box, box_idx, one_padding), dim=-1)
    bpp_obs, _, _, _ = envs.step(adv_act.cpu().numpy()[0])
    bpp_obs = bpp_obs.reshape(1, num_box + num_next_box + num_candidate_action, node_dim)

    return torch.FloatTensor(bpp_obs).to(device), box_idx


if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)