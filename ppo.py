import os, sys
import numpy as np
from time import strftime, localtime, time
from collections import deque
import random

import torch
import torch.nn as nn
import torch.optim as optim

import utils
from storage import PPO_RolloutStorage
from models.graph_attention import DRL_GAT


class PPO_Training():
    def __init__(self,
                 writer,
                 timeStr,
                 BPP_policy,
                 adv_policy,
                 mix_policy,
                 args,
                 use_clipped_value_loss=True,
                 ):

        self.BPP_policy = BPP_policy
        self.adv_policy = adv_policy
        self.mix_policy = mix_policy

        self.alpha = args.alpha
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch

        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef

        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.lr = args.learning_rate
        self.eps = args.eps

        self.writer = writer
        self.timeStr = timeStr
        self.args = args
        self.factor = args.normFactor

        self.bpp_optimizer = optim.Adam(self.BPP_policy.parameters(), lr=self.lr, eps=self.eps)
        self.adv_optimizer = optim.Adam(self.adv_policy.parameters(), lr=self.lr, eps=self.eps)
        self.mix_optimizer = optim.Adam(self.mix_policy.parameters(), lr=self.lr, eps=self.eps)

        if args.seed is not None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

    def train_n_steps(self, envs, args, device,):
        model_save_path = os.path.join(args.model_save_path, self.timeStr)
        sub_time_str = strftime('%Y.%m.%d-%H-%M-%S', localtime(time()))

        self.BPP_policy.train()
        self.adv_policy.train()
        self.mix_policy.train()

        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim

        rot_num = 2 if args.setting != 2 else 6
        batchX = self.batchX = torch.arange(num_processes).to(device)
        self.device = device

        self.bpp_rollout = PPO_RolloutStorage(
            num_steps,
            num_processes,
            obs_shape=(num_box + num_next_box + num_candidate_action, node_dim),
            action_shape=(1, ),
        )
        self.bpp_rollout.to(device)

        self.adv_rollout = PPO_RolloutStorage(
            num_steps,
            num_processes,
            obs_shape=(num_box + num_next_box, node_dim),
            action_shape=(1, ),
        )
        self.adv_rollout.to(device)

        self.mix_rollout = PPO_RolloutStorage(
            num_steps,
            num_processes,
            obs_shape=(num_box + num_next_box, node_dim),
            action_shape=(1,),
        )
        self.mix_rollout.to(device)


        self.bpp_ratio_recorder = 0
        self.bpp_episode_rewards = deque(maxlen=10)
        self.bpp_episode_ratio = deque(maxlen=10)
        self.bpp_episode_counter = deque(maxlen=10)
        self.bpp_step_counter = 1

        self.adv_ratio_recorder = 0
        self.adv_episode_rewards = deque(maxlen=10)
        self.adv_episode_ratio = deque(maxlen=10)
        self.adv_episode_counter = deque(maxlen=10)
        self.adv_step_counter = 1

        self.mix_ratio_recorder = 0
        self.mix_episode_rewards = deque(maxlen=10)
        self.mix_episode_ratio = deque(maxlen=10)
        self.mix_episode_counter = deque(maxlen=10)
        self.mix_step_counter = 1

        self.bpp_model_save_que = list()
        self.adv_model_save_que = list()
        self.mix_model_save_que = list()

        max_update_num = int(args.num_env_steps // num_steps // num_processes)
        self.bpp_start = self.adv_start = self.mix_start = time()
        while True:
            self.train_adv_policy(
                envs, model_save_path, sub_time_str, rot_num, batchX,
                max_update_num, args.adv_update_steps, args, device
            )
            self.train_mix_policy(
                envs, model_save_path, sub_time_str, rot_num, batchX,
                max_update_num, args.mix_update_steps, args, device
            )
            self.train_bpp_policy(
                envs, model_save_path, sub_time_str, rot_num, batchX,
                max_update_num, args.bpp_update_steps, args, device
            )

        return


    def train_bpp_policy(self, envs, model_save_path, sub_time_str, rot_num,
                         batchX, max_update_num, num_bpp_update, args, device):

        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim

        pmt_obs = envs.reset()
        pmt_obs = pmt_obs.reshape(pmt_obs.shape[0], num_box+num_next_box, node_dim).to(device)
        bpp_obs, box_idx = self.execute_permute_policy(envs, pmt_obs, batchX, device)
        self.bpp_rollout.obs[0].copy_(bpp_obs)

        for bpp_step in range(num_bpp_update):
            if args.use_linear_lr_decay:
                utils.update_linear_schedule(
                    self.bpp_optimizer,
                    self.bpp_step_counter - args.begin_decay_step,
                    max_update_num,
                    args.learning_rate,
                    args.minimum_lr,
                )
            self.bpp_step_counter += 1

            for step in range(num_steps):
                with torch.no_grad():
                    action_log_probs, action, entropy = self.BPP_policy.forward_actor(
                        bpp_obs, deterministic=False, normFactor=self.factor
                    )
                    value = self.BPP_policy.forward_critic(
                        bpp_obs, deterministic=False, normFactor=self.factor
                    )

                location = bpp_obs.split(
                    [num_box, num_next_box, num_candidate_action], dim=1
                )[-1][batchX, action.squeeze(1)][:, :7]
                zero_padding = torch.zeros((location.size(0), 1)).to(device)
                execution = torch.cat((location, box_idx, zero_padding), dim=-1)

                pmt_obs, reward, done, infos = envs.step(execution.cpu().numpy())
                pmt_obs = pmt_obs.reshape(pmt_obs.shape[0], num_box+num_next_box, node_dim).to(device)
                bpp_obs, box_idx = self.execute_permute_policy(envs, pmt_obs, batchX, device)

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                self.bpp_rollout.insert(bpp_obs, action, action_log_probs, value, reward, masks, bad_masks,)

                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            self.bpp_episode_rewards.append(infos[_]['reward'])
                        if 'ratio' in infos[_].keys():
                            self.bpp_episode_ratio.append(infos[_]['ratio'])
                        if 'counter' in infos[_].keys():
                            self.bpp_episode_counter.append(infos[_]['counter'])

            with torch.no_grad():
                next_value = self.BPP_policy.forward_critic(self.bpp_rollout.obs[-1], normFactor=self.factor)

            self.bpp_rollout.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits,
            )

            value_loss_epoch, action_loss_epoch, dist_entropy_epoch = self.update(
                self.bpp_rollout, self.bpp_optimizer, self.BPP_policy
            )
            self.bpp_rollout.after_update()

            self.save_model(
                self.bpp_step_counter, args.model_save_interval, args.model_update_interval,
                model_save_path, self.BPP_policy, sub_time_str,
                self.bpp_model_save_que, args.max_model_num,
                tag='BPP'
            )

            if self.bpp_step_counter % args.print_log_interval == 0 and len(self.bpp_episode_rewards) > 1:
                total_num_steps = self.bpp_step_counter * num_processes * num_steps
                end = time()

                if len(self.bpp_episode_ratio) != 0:
                    self.bpp_ratio_recorder = max(self.bpp_ratio_recorder, np.max(self.bpp_episode_ratio))

                episodes_training_results = \
                    "Train BPP policy\n" \
                    "Updates {}, num timesteps {}, FPS {}\n" \
                    "Last {} training episodes:\n" \
                    "Mean/Median Reward {:.3f}/{:.3f}, Min/Max Reward {:.3f}/{:.3f}\n" \
                    "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                    "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                    "The ratio threshold is {}\n" \
                    "The value loss {:.5f}, the action loss {:.5f}, the entropy {:.5f}\n" \
                        .format(self.bpp_step_counter, total_num_steps, int(total_num_steps / (end - self.bpp_start)),
                                len(self.bpp_episode_rewards),
                                np.mean(self.bpp_episode_rewards), np.median(self.bpp_episode_rewards),
                                np.min(self.bpp_episode_rewards), np.max(self.bpp_episode_rewards),
                                np.mean(self.bpp_episode_ratio), np.median(self.bpp_episode_ratio),
                                np.min(self.bpp_episode_ratio), np.max(self.bpp_episode_ratio),
                                np.mean(self.bpp_episode_counter), np.median(self.bpp_episode_counter),
                                np.min(self.bpp_episode_counter), np.max(self.bpp_episode_counter),
                                self.bpp_ratio_recorder,
                                value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                                )
                print(episodes_training_results)

                self.writer.add_scalar('BPP/Rewards/Mean', np.mean(self.bpp_episode_rewards), self.bpp_step_counter)
                self.writer.add_scalar("BPP/Rewards/Max", np.max(self.bpp_episode_rewards), self.bpp_step_counter)
                self.writer.add_scalar('BPP/Rewards/Min', np.min(self.bpp_episode_rewards), self.bpp_step_counter)
                self.writer.add_scalar("BPP/Ratio/Mean", np.mean(self.bpp_episode_ratio), self.bpp_step_counter)
                self.writer.add_scalar("BPP/Ratio/Max", np.max(self.bpp_episode_ratio), self.bpp_step_counter)
                self.writer.add_scalar("BPP/Ratio/Min", np.min(self.bpp_episode_ratio), self.bpp_step_counter)
                self.writer.add_scalar("BPP/Ratio/Historical Max", self.bpp_ratio_recorder, self.bpp_step_counter)
                self.writer.add_scalar('BPP/Counter/Mean', np.mean(self.bpp_episode_counter), self.bpp_step_counter)
                self.writer.add_scalar('BPP/Counter/Max', np.max(self.bpp_episode_counter), self.bpp_step_counter)
                self.writer.add_scalar('BPP/Counter/Min', np.min(self.bpp_episode_counter), self.bpp_step_counter)
                self.writer.add_scalar("BPP/Training/Value loss", value_loss_epoch, self.bpp_step_counter)
                self.writer.add_scalar("BPP/Training/Action loss", action_loss_epoch, self.bpp_step_counter)
                self.writer.add_scalar('BPP/Training/entropy', dist_entropy_epoch, self.bpp_step_counter)
                self.writer.add_scalar('BPP/Param/lr', self.bpp_optimizer.param_groups[0]['lr'], self.bpp_step_counter)

    def train_adv_policy(self, envs, model_save_path, sub_time_str, rot_num, batchX,
                         max_update_num, num_adv_update, args, device):

        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim

        adv_obs = envs.reset()
        adv_obs = adv_obs.reshape(adv_obs.shape[0], num_box+num_next_box, node_dim).to(device)
        self.adv_rollout.obs[0].copy_(adv_obs)

        for adv_step in range(num_adv_update):
            if args.use_linear_lr_decay:
                utils.update_linear_schedule(
                    self.adv_optimizer,
                    self.adv_step_counter - args.begin_decay_step,
                    max_update_num,
                    args.learning_rate,
                    args.minimum_lr
                )
            self.adv_step_counter += 1

            for step in range(num_steps):
                with torch.no_grad():
                    action_log_probs, action, entropy = self.adv_policy.forward_actor(
                        adv_obs, deterministic=False, normFactor=self.factor
                    )
                    value = self.adv_policy.forward_critic(
                        adv_obs, deterministic=False, normFactor=self.factor
                    )
                box = adv_obs.split([num_box, num_next_box, ], dim=1)[1][batchX, action.squeeze(1)][:, :7]
                one_padding = torch.ones((box.size(0), 1)).to(device)
                execution = torch.cat((box, action, one_padding), dim=-1)

                bpp_obs, _, _, _, = envs.step(execution.cpu().numpy())
                bpp_obs = bpp_obs.reshape(num_processes, num_box + num_next_box + num_candidate_action, node_dim).to(device)
                adv_obs, reward, done, infos = self.execute_bpp_policy(envs, bpp_obs, batchX, device, action)

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                self.adv_rollout.insert(adv_obs, action, action_log_probs, value, reward, masks, bad_masks)

                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            self.adv_episode_rewards.append(infos[_]['reward'])
                        if 'ratio' in infos[_].keys():
                            self.adv_episode_ratio.append(infos[_]['ratio'])
                        if 'counter' in infos[_].keys():
                            self.adv_episode_counter.append(infos[_]['counter'])

            with torch.no_grad():
                next_value = self.adv_policy.forward_critic(self.adv_rollout.obs[-1], normFactor=self.factor)

            self.adv_rollout.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits,
            )
            value_loss_epoch, action_loss_epoch, dist_entropy_epoch = self.update(
                self.adv_rollout, self.adv_optimizer, self.adv_policy
            )
            self.adv_rollout.after_update()

            self.save_model(
                self.adv_step_counter, args.model_save_interval, args.model_update_interval,
                model_save_path, self.adv_policy, sub_time_str,
                self.adv_model_save_que, args.max_model_num,
                tag='Adv'
            )

            if self.adv_step_counter % args.print_log_interval == 0 and len(self.adv_episode_rewards) > 1:
                total_num_steps = self.adv_step_counter * num_processes * num_steps
                end = time()

                if len(self.adv_episode_ratio) != 0:
                    self.adv_ratio_recorder = max(self.adv_ratio_recorder, np.max(self.adv_episode_ratio))

                episodes_training_results = \
                    "Train Adv policy\n" \
                    "Updates {}, num timesteps {}, FPS {}\n" \
                    "Last {} training episodes:\n" \
                    "Mean/Median Reward {:.3f}/{:.3f}, Min/Max Reward {:.3f}/{:.3f}\n" \
                    "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                    "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                    "The ratio threshold is {}\n" \
                    "The value loss {:.5f}, the action loss {:.5f}, the entropy {:.5f}\n" \
                        .format(self.adv_step_counter, total_num_steps, int(total_num_steps / (end - self.adv_start)),
                                len(self.adv_episode_rewards),
                                np.mean(self.adv_episode_rewards), np.median(self.adv_episode_rewards),
                                np.min(self.adv_episode_rewards), np.max(self.adv_episode_rewards),
                                np.mean(self.adv_episode_ratio), np.median(self.adv_episode_ratio),
                                np.min(self.adv_episode_ratio), np.max(self.adv_episode_ratio),
                                np.mean(self.adv_episode_counter), np.median(self.adv_episode_counter),
                                np.min(self.adv_episode_counter), np.max(self.adv_episode_counter),
                                self.adv_ratio_recorder,
                                value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                                )
                print(episodes_training_results)

                self.writer.add_scalar('Adv/Rewards/Mean', np.mean(self.adv_episode_rewards), self.adv_step_counter)
                self.writer.add_scalar("Adv/Rewards/Max", np.max(self.adv_episode_rewards), self.adv_step_counter)
                self.writer.add_scalar('Adv/Rewards/Min', np.min(self.adv_episode_rewards), self.adv_step_counter)
                self.writer.add_scalar("Adv/Ratio/Mean", np.mean(self.adv_episode_ratio), self.adv_step_counter)
                self.writer.add_scalar("Adv/Ratio/Max", np.max(self.adv_episode_ratio), self.adv_step_counter)
                self.writer.add_scalar("Adv/Ratio/Min", np.min(self.adv_episode_ratio), self.adv_step_counter)
                self.writer.add_scalar("Adv/Ratio/Historical Max", self.adv_ratio_recorder, self.adv_step_counter)
                self.writer.add_scalar('Adv/Counter/Mean', np.mean(self.adv_episode_counter), self.adv_step_counter)
                self.writer.add_scalar('Adv/Counter/Max', np.max(self.adv_episode_counter), self.adv_step_counter)
                self.writer.add_scalar('Adv/Counter/Min', np.min(self.adv_episode_counter), self.adv_step_counter)
                self.writer.add_scalar("Adv/Training/Value loss", value_loss_epoch, self.adv_step_counter)
                self.writer.add_scalar("Adv/Training/Action loss", action_loss_epoch, self.adv_step_counter)
                self.writer.add_scalar('Adv/Training/entropy', dist_entropy_epoch, self.adv_step_counter)
                self.writer.add_scalar('Adv/Param/lr', self.adv_optimizer.param_groups[0]['lr'], self.adv_step_counter)


    def train_mix_policy(self, envs, model_save_path, sub_time_str, rot_num, batchX,
                         max_update_num, num_mix_update, args, device):

        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim

        mix_obs = envs.reset()
        mix_obs = mix_obs.reshape(mix_obs.shape[0], num_box+num_next_box, node_dim).to(device)
        self.mix_rollout.obs[0].copy_(mix_obs)

        for mix_step in range(num_mix_update):
            if args.use_linear_lr_decay:
                utils.update_linear_schedule(
                    self.mix_optimizer,
                    self.mix_step_counter - args.begin_decay_step,
                    max_update_num,
                    args.learning_rate,
                    args.minimum_lr
                )
            self.mix_step_counter += 1

            for step in range(num_steps):
                with torch.no_grad():
                    action_log_probs, action, entropy = self.mix_policy.forward_actor(
                        mix_obs, deterministic=False, normFactor=self.factor
                    )
                    value = self.mix_policy.forward_critic(
                        mix_obs, deterministic=False, normFactor=self.factor
                    )
                box = mix_obs.split([num_box, num_next_box, ], dim=1)[1][batchX, action.squeeze(1)][:, :7]
                one_padding = torch.ones((box.size(0), 1)).to(device)
                execution = torch.cat((box, action, one_padding), dim=-1)

                bpp_obs, _, _, _, = envs.step(execution.cpu().numpy())
                bpp_obs = bpp_obs.reshape(num_processes, num_box + num_next_box + num_candidate_action, node_dim).to(device)
                mix_obs, reward, done, infos = self.execute_bpp_policy(envs, bpp_obs, batchX, device, action, inv_reward=False)

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                self.mix_rollout.insert(mix_obs, action, action_log_probs, value, reward, masks, bad_masks)

                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            self.mix_episode_rewards.append(infos[_]['reward'])
                        if 'ratio' in infos[_].keys():
                            self.mix_episode_ratio.append(infos[_]['ratio'])
                        if 'counter' in infos[_].keys():
                            self.mix_episode_counter.append(infos[_]['counter'])

            with torch.no_grad():
                next_value = self.mix_policy.forward_critic(self.mix_rollout.obs[-1], normFactor=self.factor)

            self.mix_rollout.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits,
            )

            value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
            distance_loss_epoch, normal_loss_epoch, worst_loss_epoch = self.update(
                self.mix_rollout, self.mix_optimizer, self.mix_policy, dist_loss=True
            )
            self.mix_rollout.after_update()

            self.save_model(
                self.mix_step_counter, args.model_save_interval, args.model_update_interval,
                model_save_path, self.mix_policy, sub_time_str,
                self.mix_model_save_que, args.max_model_num,
                tag='Mix'
            )

            if self.mix_step_counter % args.print_log_interval == 0 and len(self.mix_episode_rewards) > 1:
                total_num_steps = self.mix_step_counter * num_processes * num_steps
                end = time()

                if len(self.mix_episode_ratio) != 0:
                    self.mix_ratio_recorder = max(self.mix_ratio_recorder, np.max(self.mix_episode_ratio))

                episodes_training_results = \
                    "Train Mix policy\n" \
                    "Updates {}, num timesteps {}, FPS {}\n" \
                    "Last {} training episodes:\n" \
                    "Mean/Median Reward {:.3f}/{:.3f}, Min/Max Reward {:.3f}/{:.3f}\n" \
                    "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                    "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                    "The ratio threshold is {}\n" \
                    "The value loss {:.5f}, the action loss {:.5f}, the entropy {:.5f}\n" \
                    "The distance loss {:.5f}, the normal loss {:.5f}, the worst loss {:.5f}\n" \
                        .format(self.mix_step_counter, total_num_steps, int(total_num_steps / (end - self.mix_start)),
                                len(self.mix_episode_rewards),
                                np.mean(self.mix_episode_rewards), np.median(self.mix_episode_rewards),
                                np.min(self.mix_episode_rewards), np.max(self.mix_episode_rewards),
                                np.mean(self.mix_episode_ratio), np.median(self.mix_episode_ratio),
                                np.min(self.mix_episode_ratio), np.max(self.mix_episode_ratio),
                                np.mean(self.mix_episode_counter), np.median(self.mix_episode_counter),
                                np.min(self.mix_episode_counter), np.max(self.mix_episode_counter),
                                self.mix_ratio_recorder,
                                value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                                distance_loss_epoch, normal_loss_epoch, worst_loss_epoch,
                                )
                print(episodes_training_results)

                self.writer.add_scalar('Mix/Rewards/Mean', np.mean(self.mix_episode_rewards), self.mix_step_counter)
                self.writer.add_scalar("Mix/Rewards/Max", np.max(self.mix_episode_rewards), self.mix_step_counter)
                self.writer.add_scalar('Mix/Rewards/Min', np.min(self.mix_episode_rewards), self.mix_step_counter)
                self.writer.add_scalar("Mix/Ratio/Mean", np.mean(self.mix_episode_ratio), self.mix_step_counter)
                self.writer.add_scalar("Mix/Ratio/Max", np.max(self.mix_episode_ratio), self.mix_step_counter)
                self.writer.add_scalar("Mix/Ratio/Min", np.min(self.mix_episode_ratio), self.mix_step_counter)
                self.writer.add_scalar("Mix/Ratio/Historical Max", self.mix_ratio_recorder, self.mix_step_counter)
                self.writer.add_scalar('Mix/Counter/Mean', np.mean(self.mix_episode_counter), self.mix_step_counter)
                self.writer.add_scalar('Mix/Counter/Max', np.max(self.mix_episode_counter), self.mix_step_counter)
                self.writer.add_scalar('Mix/Counter/Min', np.min(self.mix_episode_counter), self.mix_step_counter)
                self.writer.add_scalar("Mix/Training/Value loss", value_loss_epoch, self.mix_step_counter)
                self.writer.add_scalar("Mix/Training/Action loss", action_loss_epoch, self.mix_step_counter)
                self.writer.add_scalar('Mix/Training/entropy', dist_entropy_epoch, self.mix_step_counter)
                self.writer.add_scalar('Mix/Param/lr', self.mix_optimizer.param_groups[0]['lr'], self.mix_step_counter)
                self.writer.add_scalar('Mix/Dist/Sum', distance_loss_epoch, self.mix_step_counter)
                self.writer.add_scalar('Mix/Dist/Normal', normal_loss_epoch, self.mix_step_counter)
                self.writer.add_scalar('Mix/Dist/Worst', worst_loss_epoch, self.mix_step_counter)
    

    def execute_bpp_policy(self, envs, bpp_obs, batchX, device, box_idx, inv_reward=True):
        num_box, num_next_box, num_candidate_action = \
            self.args.num_box, self.args.num_next_box, self.args.num_candidate_action
        with torch.no_grad():
            _, loc_idx, _, = self.BPP_policy.forward_actor(
                bpp_obs, deterministic=False, normFactor=self.factor,
            )
        tmp_loc = bpp_obs.split([num_box, num_next_box, num_candidate_action], dim=1)[-1][batchX, loc_idx.squeeze(1)][:, :7]
        zero_padding = torch.zeros((tmp_loc.size(0), 1)).to(device)
        bpp_act = torch.cat((tmp_loc, box_idx, zero_padding), dim=-1)
        obs, reward, done, infos = envs.step(bpp_act.cpu().numpy())
        reward = -reward if inv_reward else reward
        return obs.reshape(obs.size(0), num_box + num_next_box, -1).to(device), reward, done, infos


    def execute_permute_policy(self, envs, pmt_obs, batchX, device):
        num_box, num_next_box, num_candidate_action = \
            self.args.num_box, self.args.num_next_box, self.args.num_candidate_action
        with torch.no_grad():
            _, box_idx, _, = self.mix_policy.forward_actor(pmt_obs, deterministic=False, normFactor=self.factor)
        tmp_box = pmt_obs.split([num_box, num_next_box, ], dim=1)[1][batchX, box_idx.squeeze(1)][:, :7]
        one_padding = torch.ones((tmp_box.size(0), 1)).to(device)
        pmt_act = torch.cat((tmp_box, box_idx, one_padding), dim=-1)
        bpp_obs, _, _, _, = envs.step(pmt_act.cpu().numpy())

        return bpp_obs.reshape(bpp_obs.size(0), num_box + num_next_box + num_candidate_action, -1).to(device), box_idx


    def save_model(self, step_counter, save_interval, update_interval, model_save_path,
                   model, sub_time_str, model_save_que, max_model_num, tag):

        if step_counter % save_interval == 0 and model_save_path != "":
            if step_counter % update_interval == 0:
                sub_time_str = strftime('%Y.%m.%d-%H-%M-%S', localtime(time()))

            if sub_time_str not in model_save_que:
                model_save_que.append(sub_time_str)

            if len(model_save_que) > max_model_num:
                rm_model = model_save_que.pop(0)
                os.remove(os.path.join(model_save_path, '{}-{}.pt'.format(tag, rm_model)))

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(
                model.state_dict(),
                os.path.join(model_save_path, '{}-{}.pt'.format(tag, sub_time_str))
            )
        return

    def update(self,
               rollouts: PPO_RolloutStorage,
               optimizer: torch.optim.Optimizer,
               policy_model: DRL_GAT,
               dist_loss=False,
               ):

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        distance_loss_epoch = 0
        normal_loss_epoch = 0
        worst_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if policy_model.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, old_action_log_probs_batch, adv_targ = sample

                action_log_probs, dist_entropy, = policy_model.evaluate_actions(
                    obs_batch, actions_batch, normFactor=self.factor
                )
                values = policy_model.evaluate_values(
                    obs_batch, normFactor=self.factor
                )

                if dist_loss:
                    # distance between normal transition and mixture transition
                    normal_action = torch.zeros((obs_batch.size(0), 1)).to(self.device)
                    normal_action_log_probs, _ = policy_model.evaluate_actions(
                        obs_batch, normal_action, normFactor=self.factor
                    )
                    normal_loss = (-normal_action_log_probs).mean()

                    mix_log_probs = policy_model.action_log_probs(obs_batch, normFactor=self.factor)
                    with torch.no_grad():
                        wor_log_probs = self.adv_policy.action_log_probs(obs_batch, normFactor=self.factor)

                    kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
                    worst_loss = kl_loss(mix_log_probs, wor_log_probs)

                    distance_loss = normal_loss + self.alpha * worst_loss
                else:
                    normal_loss = torch.zeros(1, requires_grad=True).to(self.device)
                    worst_loss = torch.zeros(1, requires_grad=True).to(self.device)
                    distance_loss = torch.zeros(1, requires_grad=True).to(self.device)


                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                optimizer.zero_grad()
                (value_loss * self.value_loss_coef
                 + action_loss
                 - dist_entropy * self.entropy_coef
                 + distance_loss
                 ).backward()
                nn.utils.clip_grad_norm_(policy_model.parameters(), self.max_grad_norm)
                optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                distance_loss_epoch += distance_loss.item()
                normal_loss_epoch += normal_loss.item()
                worst_loss_epoch += worst_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        distance_loss_epoch /= num_updates
        normal_loss_epoch /= num_updates
        worst_loss_epoch /= num_updates

        if distance_loss:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
                   distance_loss_epoch, normal_loss_epoch, worst_loss_epoch
        else:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch,