import os, sys
import time
from models.graph_attention import DRL_GAT
from utils import registration_envs, get_args, load_ppo_policy, Logger
import gym
import torch
from time import strftime, localtime, time
import numpy as np
from pprint import pprint


def evaluate_one_dist(BPP_policy, adv_policy, eval_envs, timeStr, args, device, eval_freq=100, factor=1):
    # Save the test trajectories.
    if not os.path.exists('{}/evaluation/'.format(args.log_path) + timeStr):
        os.makedirs('{}/evaluation/'.format(args.log_path) + timeStr)

    num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
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
            action_log_probs, action, entropy = BPP_policy.forward_actor(
                bpp_obs, deterministic=True, normFactor=factor
            )
            value = BPP_policy.forward_critic(
                bpp_obs, deterministic=True, normFactor=factor
            )

        location = bpp_obs.split([num_box + num_next_box, num_candidate_action], dim=1)[-1][[0], action.squeeze(1)][:, :7]
        zero_padding = torch.zeros((location.size(0), 1)).to(device)
        execution = torch.cat((location, box_idx, zero_padding), dim=-1)

        adv_obs, reward, done, infos = eval_envs.step(execution[0].cpu().numpy())

        if done:
            if 'ratio' in infos.keys():
                episode_ratio.append(infos['ratio'])
            if 'counter' in infos.keys():
                episode_length.append(infos['counter'])

            one_episode_result = 'Episode: {} \n' \
                                 'Mean ratio: {}, length: {}\n' \
                                 'Episode ratio: {}, length: {}\n\n' \
                                 .format(step_counter,
                                         np.mean(episode_ratio), np.mean(episode_length),
                                         infos['ratio'], infos['counter'])
            print(one_episode_result)

            items = [
                (item.x, item.y, item.z, item.w, item.h, item.l, item.density) for item in eval_envs.space.boxes[1:]
            ]
            all_episodes.append(items)
            step_counter += 1
            adv_obs = eval_envs.reset()

        adv_obs = adv_obs.reshape(1, num_box + num_next_box, args.node_dim)
        adv_obs = torch.FloatTensor(adv_obs).to(device)
        bpp_obs, box_idx = execute_permute_policy(adv_policy, eval_envs, adv_obs, device, args)

    print(
        "Evaluation using {} episodes\n" \
        "Mean ratio {:.5f}, mean length {:.5f}\n". \
            format(len(episode_ratio), np.mean(episode_ratio), np.mean(episode_length))
    )
    np.save(os.path.join('{}/evaluation'.format(args.log_path), timeStr, 'trajs.npy'), all_episodes)

    return


def execute_permute_policy(adv_policy, envs, adv_obs, device, args):
    num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
    node_dim = args.node_dim

    if args.load_adv_model:
        with torch.no_grad():
            _, box_idx, _, = adv_policy.forward_actor(adv_obs, deterministic=True, normFactor=args.normFactor)
    else:
        box_idx = torch.zeros(adv_obs.size(0))[:, None].to(device).to(torch.long)

    tmp_box = adv_obs.split([num_box, num_next_box, ], dim=1)[1][[0], box_idx.squeeze(1)][:, :7]
    one_padding = torch.ones((tmp_box.size(0), 1)).to(device)
    adv_act = torch.cat((tmp_box, box_idx, one_padding), dim=-1)
    bpp_obs, _, _, _, = envs.step(adv_act.cpu().numpy()[0])

    bpp_obs = bpp_obs.reshape(1, num_box + num_next_box + num_candidate_action, node_dim)
    return torch.FloatTensor(bpp_obs).to(device), box_idx



def main(args):
    box_setting = 'continuous' if args.continuous else 'discrete'
    timeStr = args.training_algorithm + '_setting{}_{}_'.format(args.setting, box_setting) + \
              strftime('%Y.%m.%d-%H-%M-%S', localtime(time())) + '_{}'.format(round(np.random.rand(), 5))

    os.makedirs('{}/logger/test'.format(args.log_path), exist_ok=True)
    sys.stdout = Logger('{}/logger/test/{}.log'.format(args.log_path, timeStr), sys.stdout)
    sys.stderr = Logger('{}/logger/test/{}.log'.format(args.log_path, timeStr), sys.stderr)
    pprint(vars(args))

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create single packing environment and load existing dataset.
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

    BPP_policy = BPP_policy.to(device)
    adv_policy = adv_policy.to(device)


    if args.load_bpp_model:
        BPP_policy = load_ppo_policy(args.bpp_model_path, BPP_policy)
    if args.load_adv_model:
        adv_policy = load_ppo_policy(args.adv_model_path, adv_policy)

    evaluate_one_dist(
        BPP_policy, adv_policy, envs, timeStr, args, device,
        eval_freq=args.evaluation_episodes, factor=args.normFactor,
    )


if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)