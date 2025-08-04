import os, sys
import random
import numpy as np
import pprint
import copy
from time import strftime, localtime, time
from tensorboardX import SummaryWriter
import gym
import torch
import wandb

from envs import make_vec_envs
from models.graph_attention import DRL_GAT
from ppo import PPO_Training
from utils import get_args, registration_envs, load_ppo_policy, Logger


def main(args):
    timeStr = f"{strftime('%Y.%m.%d-%H-%M-%S', localtime(time()))}_{str(round(np.random.rand(), 5))}"
    experiment = f"AR2L_Training_{args.training_algorithm}_setting{args.setting}_" \
                 f"{('continuous' if args.continuous else 'discrete')}_nnb{args.num_next_box}_{timeStr}"

    args.project_path = os.path.join(args.log_path, 'train', experiment)
    os.makedirs(args.project_path, exist_ok=True)

    sys.stdout = Logger(f'{args.project_path}/train.log', sys.stdout)
    sys.stderr = Logger(f'{args.project_path}/train.log', sys.stderr)

    configs = copy.deepcopy(vars(args))
    configs.pop('item_size_set')
    pprint.pprint(configs)

    # TODO: please use your personal account
    run_exp = None
    if args.use_wandb:
        wandb.login(key="")
        run_exp = wandb.init()

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

    envs = make_vec_envs(args, None, True)

    bppObs_size = (args.num_box + args.num_next_box + args.num_candidate_action) * args.node_dim
    bpp_policy = DRL_GAT(gym.spaces.Box(low=0., high=args.container_size[-1], shape=(bppObs_size, )),
                         gym.spaces.Discrete(n=args.num_candidate_action),
                         args.embedding_size,
                         args.hidden_size,
                         args.gat_layer_num,
                         args.num_box,
                         args.num_next_box,
                         args.num_candidate_action,
                         args.node_dim,
                         policy_ctg='placement',
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

    bpp_policy = bpp_policy.to(device)
    adv_policy = adv_policy.to(device)
    mix_policy = mix_policy.to(device)

    if args.load_bpp_model:
        bpp_policy = load_ppo_policy(args.bpp_model_path, bpp_policy)
    if args.load_adv_model:
        adv_policy = load_ppo_policy(args.adv_model_path, adv_policy)
    if args.load_mix_model:
        bal_policy = load_ppo_policy(args.mix_model_path, mix_policy)


    train_model = PPO_Training(
        bpp_policy,
        adv_policy,
        mix_policy,
        device,
        experiment,
        args,
        run_exp,
    )
    train_model.train_n_steps(envs)

    return


if __name__ == '__main__':
    registration_envs()
    args = get_args()
    main(args)


