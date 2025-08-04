import os, sys
import random
import numpy as np
import pprint
import copy
from time import strftime, localtime, time
import gym
import torch


from envs import make_vec_envs
from models.graph_attention import DRL_GAT
from utils import registration_envs, get_args, load_ppo_policy, Logger



def evaluate_func(bpp_policy, adv_policy, envs, device, args, num_episodes=100, use_adv=False):
    num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
    node_dim = args.node_dim
    factor = args.normFactor
    num_processes = args.num_processes

    bpp_policy.eval()
    adv_policy.eval()

    batchX = torch.arange(num_processes).to(device)

    assigned_num_traj = np.full((num_processes,), dtype=np.int32, fill_value=int(num_episodes // num_processes))
    assigned_num_traj[:int(num_episodes % num_processes)] += 1

    assert num_episodes >= num_processes
    assert assigned_num_traj.sum() == num_episodes

    start_time = time()
    adv_obs = envs.reset()
    adv_obs = adv_obs.reshape(adv_obs.size(0), num_box + num_next_box, args.node_dim).to(device)
    bpp_obs = execute_permute_policy(adv_policy, envs, adv_obs, device, batchX, args, use_adv=use_adv)

    episode_counter = 0
    episode_ratio = [[] for _ in range(num_processes)]
    episode_length = [[] for _ in range(num_processes)]
    all_episodes = [[] for _ in range(num_processes)]

    while episode_counter < num_episodes:
        with torch.no_grad():
            _, action, _ = bpp_policy.forward_actor(bpp_obs, deterministic=True, normFactor=factor)

        location = bpp_obs.split([num_box + num_next_box, num_candidate_action], dim=1)[-1][batchX, action.squeeze(1)][:, :7]
        padding = torch.zeros((location.size(0), 1)).to(device)
        execution = torch.cat((location, padding, padding), dim=-1)

        adv_obs, reward, done, infos = envs.step(execution.cpu().numpy())
        adv_obs = adv_obs.reshape(adv_obs.size(0), num_box + num_next_box, node_dim).to(device)

        for _ in range(len(infos)):
            if done[_]:
                if len(episode_ratio[_]) < assigned_num_traj[_]:
                    episode_ratio[_].append(infos[_]['ratio'])
                    episode_length[_].append(infos[_]['counter'])
                    all_episodes[_].append(infos[_]['packed_box'])
                    episode_counter += 1

        bpp_obs = execute_permute_policy(adv_policy, envs, adv_obs, device, batchX, args, use_adv=use_adv)

    total_time = time() - start_time
    envs.close()

    episode_ratio = sum(episode_ratio, [])
    episode_length = sum(episode_length, [])
    all_episodes = sum(all_episodes, [])
    assert len(episode_ratio) == len(episode_length) == len(all_episodes) == num_episodes

    return episode_ratio, episode_length, total_time, all_episodes





def execute_permute_policy(adv_policy, envs, adv_obs, device, batchX, args, use_adv=False):
    num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
    node_dim = args.node_dim

    if use_adv:
        with torch.no_grad():
            _, box_idx, _, = adv_policy.forward_actor(adv_obs, deterministic=True, normFactor=args.normFactor)
    else:
        box_idx = torch.zeros_like(batchX)[:, None].to(device)

    tmp_box = adv_obs.split([num_box, num_next_box], dim=1)[1][batchX, box_idx.squeeze(1)][:, :7]
    padding = torch.ones((tmp_box.size(0), 1)).to(device)
    adv_act = torch.cat((tmp_box, box_idx, padding), dim=-1)
    bpp_obs, _, _, _, = envs.step(adv_act.cpu().numpy())

    bpp_obs = bpp_obs.reshape(bpp_obs.size(0), num_box + num_next_box + num_candidate_action, node_dim).to(device)
    return bpp_obs




def main(args):

    timeStr = f"{strftime('%Y.%m.%d-%H-%M-%S', localtime(time()))}_{str(round(np.random.rand(), 5))}"
    experiment = f"AR2L_Evaluation_{args.training_algorithm}_setting{args.setting}_" \
                 f"{('continuous' if args.continuous else 'discrete')}_nnb{args.num_next_box}_{timeStr}"

    args.project_path = os.path.join(args.log_path, 'eval', experiment)
    os.makedirs(args.project_path, exist_ok=True)

    sys.stdout = Logger(f'{args.project_path}/eval.log', sys.stdout)
    sys.stderr = Logger(f'{args.project_path}/eval.log', sys.stderr)

    configs = copy.deepcopy(vars(args))
    configs.pop('item_size_set')
    pprint.pprint(configs)

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)
        torch.cuda.set_device(args.device)

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    envs = make_vec_envs(args, None, True)

    bppObs_size = (args.num_box + args.num_next_box + args.num_candidate_action) * args.node_dim
    bpp_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(bppObs_size,)),
                         gym.spaces.Discrete(n=args.num_candidate_action),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         args.num_candidate_action,
                         args.node_dim,
                         policy_ctg='placement'
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

    bpp_policy = bpp_policy.to(device)
    adv_policy = adv_policy.to(device)

    if args.load_bpp_model:
        bpp_policy = load_ppo_policy(args.bpp_model_path, bpp_policy)
    if args.load_adv_model:
        adv_policy = load_ppo_policy(args.adv_model_path, adv_policy)

    episode_ratio, episode_length, total_time, all_episodes = evaluate_func(
        bpp_policy, adv_policy, envs, device, args, num_episodes=args.num_eval_episodes, use_adv=args.load_adv_model
    )

    print(
        f"Evaluation using {args.num_eval_episodes} episodes\n" \
        f"Mean ratio {np.mean(episode_ratio):.5f}+-{np.std(episode_ratio)}, Mean length {np.mean(episode_length):.5f}\n" \
        f"Total time(s) {total_time:.3f}, Average time(s) {(total_time / args.num_eval_episodes):.3f}\n" \
        )
    np.save(os.path.join(args.project_path, 'trajs.npy'), np.array(all_episodes, dtype=object))


if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)