import os, sys
import numpy as np
from time import strftime, localtime, time
from collections import deque
import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import utils
from storage import PPO_RolloutStorage
from models.graph_attention import DRL_GAT
from evaluation import evaluate_func
from envs import make_vec_envs


class PPO_Training():
    def __init__(self,
                 bpp_policy,
                 adv_policy,
                 mix_policy,
                 device,
                 experiment,
                 args,
                 run_exp,
                 use_clipped_value_loss=True,
                 ):

        self.bpp_policy = bpp_policy
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

        self.args = args
        self.device = device
        self.experiment = experiment
        self.run_exp = run_exp

        self.factor = args.normFactor
        self.batchX = torch.arange(args.num_processes).to(device)

        self.bpp_optimizer = optim.Adam(self.bpp_policy.parameters(), lr=self.lr, eps=self.eps)
        self.adv_optimizer = optim.Adam(self.adv_policy.parameters(), lr=self.lr, eps=self.eps)
        self.mix_optimizer = optim.Adam(self.mix_policy.parameters(), lr=self.lr, eps=self.eps)

        if args.seed is not None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)

    def train_n_steps(self, envs):
        args = self.args
        device = self.device
        batchX = self.batchX

        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim
        rot_num = 2 if args.setting != 2 else 6

        self.bpp_policy.train()
        self.adv_policy.train()
        self.mix_policy.train()

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
        self.bpp_step_counter = 0

        self.adv_ratio_recorder = 0
        self.adv_episode_rewards = deque(maxlen=10)
        self.adv_episode_ratio = deque(maxlen=10)
        self.adv_episode_counter = deque(maxlen=10)
        self.adv_step_counter = 0

        self.mix_ratio_recorder = 0
        self.mix_episode_rewards = deque(maxlen=10)
        self.mix_episode_ratio = deque(maxlen=10)
        self.mix_episode_counter = deque(maxlen=10)
        self.mix_step_counter = 0


        max_update_num = int(args.num_env_steps // num_steps // num_processes)
        self.bpp_start = self.adv_start = self.mix_start = time()
        while True:
            self.train_adv_policy(envs, rot_num, batchX, max_update_num, args.adv_update_steps, args, device)
            self.train_mix_policy(envs, rot_num, batchX, max_update_num, args.mix_update_steps, args, device)
            self.train_bpp_policy(envs, rot_num, batchX, max_update_num, args.bpp_update_steps, args, device)
        return


    def train_bpp_policy(self, envs, rot_num, batchX, max_update_num, num_bpp_update, args, device):
        num_processes, num_steps = args.num_processes, args.num_steps
        num_box, num_next_box, num_candidate_action = args.num_box, args.num_next_box, args.num_candidate_action
        node_dim = args.node_dim

        pmt_obs = envs.reset()
        pmt_obs = pmt_obs.reshape(pmt_obs.shape[0], num_box+num_next_box, node_dim).to(device)
        bpp_obs = self.execute_permute_policy(envs, pmt_obs)

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
                    action_log_probs, action, entropy = self.bpp_policy.forward_actor(
                        bpp_obs, deterministic=False, normFactor=self.factor
                    )
                    value = self.bpp_policy.forward_critic(
                        bpp_obs, deterministic=False, normFactor=self.factor
                    )

                location = bpp_obs.split([num_box, num_next_box, num_candidate_action], dim=1)[-1][batchX, action.squeeze(1)][:, :7]
                zero_padding = torch.zeros((location.size(0), 1)).to(device)
                execution = torch.cat((location, zero_padding, zero_padding), dim=-1)

                pmt_obs, reward, done, infos = envs.step(execution.cpu().numpy())
                pmt_obs = pmt_obs.reshape(pmt_obs.shape[0], num_box+num_next_box, node_dim).to(device)
                bpp_obs = self.execute_permute_policy(envs, pmt_obs)

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                self.bpp_rollout.insert(bpp_obs, action, action_log_probs, value, reward, masks, bad_masks,)

                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            self.bpp_episode_rewards.append(infos[_]['reward'])
                        else:
                            self.bpp_episode_rewards.append(infos[_]['episode']['r'])
                        if 'ratio' in infos[_].keys():
                            self.bpp_episode_ratio.append(infos[_]['ratio'])
                        if 'counter' in infos[_].keys():
                            self.bpp_episode_counter.append(infos[_]['counter'])

            with torch.no_grad():
                next_value = self.bpp_policy.forward_critic(self.bpp_rollout.obs[-1], normFactor=self.factor)

            self.bpp_rollout.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits,
            )

            value_loss_epoch, action_loss_epoch, dist_entropy_epoch = self.update(
                self.bpp_rollout, self.bpp_optimizer, self.bpp_policy
            )

            self.bpp_rollout.after_update()

            total_env_step = self.bpp_step_counter * num_processes * num_steps
            if len(self.bpp_episode_ratio) != 0:
                self.bpp_ratio_recorder = max(self.bpp_ratio_recorder, np.max(self.bpp_episode_ratio))

            self.save_model(
                self.bpp_step_counter, args.model_save_interval, self.bpp_policy, tag='bpp'
            )
            self.log_training(
                self.bpp_episode_rewards, self.bpp_episode_ratio, self.bpp_episode_counter, self.bpp_ratio_recorder,
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch, self.bpp_optimizer.param_groups[0]['lr'],
                self.bpp_step_counter, total_env_step, tag='bpp'
            )
            self.print_training(
                self.bpp_episode_rewards, self.bpp_episode_ratio, self.bpp_episode_counter, self.bpp_ratio_recorder,
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                self.bpp_step_counter, total_env_step, self.bpp_start, tag='bpp'
            )

            self.validation(self.bpp_step_counter, total_env_step, args.validate_interval)



    def train_adv_policy(self, envs, rot_num, batchX, max_update_num, num_adv_update, args, device):
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
                adv_obs, reward, done, infos = self.execute_bpp_policy(envs, bpp_obs)

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                self.adv_rollout.insert(adv_obs, action, action_log_probs, value, reward, masks, bad_masks)

                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            self.adv_episode_rewards.append(infos[_]['reward'])
                        else:
                            self.adv_episode_rewards.append(infos[_]['episode']['r'])
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

            total_env_step = self.adv_step_counter * num_processes * num_steps
            if len(self.adv_episode_ratio) != 0:
                self.adv_ratio_recorder = max(self.adv_ratio_recorder, np.max(self.adv_episode_ratio))

            self.save_model(
                self.adv_step_counter, args.model_save_interval, self.adv_policy, tag='adv'
            )
            self.log_training(
                self.adv_episode_rewards, self.adv_episode_ratio, self.adv_episode_counter, self.adv_ratio_recorder,
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch, self.adv_optimizer.param_groups[0]['lr'],
                self.adv_step_counter, total_env_step, tag='adv'
            )
            self.print_training(
                self.adv_episode_rewards, self.adv_episode_ratio, self.adv_episode_counter, self.adv_ratio_recorder,
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                self.adv_step_counter, total_env_step, self.adv_start, tag='adv'
            )


    def train_mix_policy(self, envs, rot_num, batchX, max_update_num, num_mix_update, args, device):
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
                mix_obs, reward, done, infos = self.execute_bpp_policy(envs, bpp_obs, inv_reward=False)

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                self.mix_rollout.insert(mix_obs, action, action_log_probs, value, reward, masks, bad_masks)

                for _ in range(len(infos)):
                    if done[_]:
                        if 'reward' in infos[_].keys():
                            self.mix_episode_rewards.append(infos[_]['reward'])
                        else:
                            self.mix_episode_rewards.append(infos[_]['episode']['r'])
                        if 'ratio' in infos[_].keys():
                            self.mix_episode_ratio.append(infos[_]['ratio'])
                        if 'counter' in infos[_].keys():
                            self.mix_episode_counter.append(infos[_]['counter'])

            with torch.no_grad():
                next_value = self.mix_policy.forward_critic(self.mix_rollout.obs[-1], normFactor=self.factor)

            self.mix_rollout.compute_returns(
                next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits,
            )

            value_loss_epoch, action_loss_epoch, dist_entropy_epoch, distance_loss_epoch_tuple = self.update(
                self.mix_rollout, self.mix_optimizer, self.mix_policy, dist_loss=True
            )
            self.mix_rollout.after_update()

            total_env_step = self.mix_step_counter * num_processes * num_steps
            if len(self.mix_episode_ratio) != 0:
                self.mix_ratio_recorder = max(self.mix_ratio_recorder, np.max(self.mix_episode_ratio))

            self.save_model(
                self.mix_step_counter, args.model_save_interval, self.mix_policy, tag='mix'
            )
            self.log_training(
                self.mix_episode_rewards, self.mix_episode_ratio, self.mix_episode_counter, self.mix_ratio_recorder,
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch, self.mix_optimizer.param_groups[0]['lr'],
                self.mix_step_counter, total_env_step, tag='mix', distance_loss_epoch_tuple=distance_loss_epoch_tuple
            )
            self.print_training(
                self.mix_episode_rewards, self.mix_episode_ratio, self.mix_episode_counter, self.mix_ratio_recorder,
                value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                self.mix_step_counter, total_env_step, self.mix_start, tag='mix', distance_loss_epoch_tuple=distance_loss_epoch_tuple
            )


    def execute_bpp_policy(self, envs, bpp_obs, inv_reward=True):
        num_box, num_next_box, num_candidate_action = \
            self.args.num_box, self.args.num_next_box, self.args.num_candidate_action
        batchX, device = self.batchX, self.device

        with torch.no_grad():
            _, loc_idx, _, = self.bpp_policy.forward_actor(
                bpp_obs, deterministic=False, normFactor=self.factor,
            )

        tmp_loc = bpp_obs.split([num_box, num_next_box, num_candidate_action], dim=1)[-1][batchX, loc_idx.squeeze(1)][:, :7]
        zero_padding = torch.zeros((tmp_loc.size(0), 1)).to(device)
        bpp_act = torch.cat((tmp_loc, zero_padding, zero_padding), dim=-1)
        obs, reward, done, infos = envs.step(bpp_act.cpu().numpy())

        reward = -reward if inv_reward else reward
        return obs.reshape(obs.size(0), num_box + num_next_box, -1).to(device), reward, done, infos


    def execute_permute_policy(self, envs, pmt_obs):
        num_box, num_next_box, num_candidate_action = \
            self.args.num_box, self.args.num_next_box, self.args.num_candidate_action
        batchX, device = self.batchX, self.device

        with torch.no_grad():
            _, box_idx, _, = self.mix_policy.forward_actor(pmt_obs, deterministic=False, normFactor=self.factor)

        tmp_box = pmt_obs.split([num_box, num_next_box, ], dim=1)[1][batchX, box_idx.squeeze(1)][:, :7]
        one_padding = torch.ones((tmp_box.size(0), 1)).to(device)
        pmt_act = torch.cat((tmp_box, box_idx, one_padding), dim=-1)
        bpp_obs, _, _, _, = envs.step(pmt_act.cpu().numpy())

        return bpp_obs.reshape(bpp_obs.size(0), num_box + num_next_box + num_candidate_action, -1).to(device)


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

                # Reshape to do in a single forward pass for all steps
                action_log_probs, dist_entropy, = policy_model.evaluate_actions(
                    obs_batch, actions_batch, normFactor=self.factor
                )
                values = policy_model.evaluate_values(
                    obs_batch, normFactor=self.factor
                )

                if dist_loss:
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
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, (distance_loss_epoch, normal_loss_epoch, worst_loss_epoch)
        else:
            return value_loss_epoch, action_loss_epoch, dist_entropy_epoch,



    def save_model(self, step_counter, save_interval, model, tag):
        if step_counter % save_interval == 0:
            sub_time_str = strftime('%Y.%m.%d-%H-%M-%S', localtime(time()))
            torch.save(
                model.state_dict(),
                os.path.join(self.args.project_path, f'{tag}-{sub_time_str}.pt')
            )
        return


    def validation(self, update_step, env_step, validate_interval):
        if update_step % validate_interval == 0 and self.args.validate:
            args = self.args
            device = self.device
            num_val_episodes = args.num_val_episodes
            assert args.dataset_path != None

            envs = make_vec_envs(args, None, True, validate=True)
            episode_ratio1, episode_length1, total_time1, _ = evaluate_func(
                self.bpp_policy, self.adv_policy, envs, device, args, num_episodes=num_val_episodes, use_adv=False
            )

            envs = make_vec_envs(args, None, True, validate=True)
            episode_ratio2, episode_length2, total_time2, _ = evaluate_func(
                self.bpp_policy, self.adv_policy, envs, device, args, num_episodes=num_val_episodes, use_adv=True
            )

            print(
                f"Normal-Case Validation on {num_val_episodes} episodes: Ratio {np.mean(episode_ratio1):.5f}, Length {np.mean(episode_length1):.5f}, Time(s): {total_time1/num_val_episodes:.3f}\n" \
                f"Worst-Case Validation on {num_val_episodes} episodes: Ratio {np.mean(episode_ratio2):.5f}, Length {np.mean(episode_length2):.5f}, Time(s): {total_time2/num_val_episodes:.3f}\n" \
            )

            self.log_validation(np.mean(episode_ratio1), np.mean(episode_length1), total_time1/num_val_episodes, update_step, env_step, tag='normal')
            self.log_validation(np.mean(episode_ratio2), np.mean(episode_length2), total_time2/num_val_episodes, update_step, env_step, tag='worst')

        return


    def log_validation(self, avg_ratio, avg_length, avg_time, update_step, env_step, tag):
        if self.run_exp != None:
            wandb_dict = {
                f"validation/{tag}/avg_ratio": avg_ratio,
                f"validation/{tag}/avg_length": avg_length,
                f"validation/{tag}/avg_time": avg_time,
                f"validation/{tag}/update_step": update_step,
                f"validation/{tag}/env_step": env_step,
            }
            self.run_exp.log(
                wandb_dict,
            )
        return


    def log_training(self, episode_rewards, episode_ratio, episode_counter, ratio_recorder,
                     value_loss_epoch, action_loss_epoch, dist_entropy_epoch, lr, update_step, env_step, tag,
                     distance_loss_epoch_tuple=None,):

        if self.run_exp != None:
            wandb_dict = {
                    f"training/{tag}/reward/mean": np.mean(episode_rewards),
                    f"training/{tag}/reward/max": np.max(episode_rewards),
                    f"training/{tag}/reward/min": np.min(episode_rewards),
                    f"training/{tag}/ratio/mean": np.mean(episode_ratio),
                    f"training/{tag}/ratio/max": np.max(episode_ratio),
                    f"training/{tag}/ratio/min": np.min(episode_ratio),
                    f"training/{tag}/ratio/historical_max": ratio_recorder,
                    f"training/{tag}/counter/mean": np.mean(episode_counter),
                    f"training/{tag}/counter/max": np.max(episode_counter),
                    f"training/{tag}/counter/min": np.min(episode_counter),
                    f"training/{tag}/value_loss": value_loss_epoch,
                    f"training/{tag}/action_loss": action_loss_epoch,
                    f"training/{tag}/entropy": dist_entropy_epoch,
                    f"training/{tag}/lr": lr,
                    f"training/{tag}/update_step": update_step,
                    f"training/{tag}/env_step": env_step,
                }
            if distance_loss_epoch_tuple != None:
                wandb_dict[f"training/{tag}/distance_loss"] = distance_loss_epoch_tuple[0]
                wandb_dict[f"training/{tag}/normal_distance_loss"] = distance_loss_epoch_tuple[1]
                wandb_dict[f"training/{tag}/worst_distance_loss"] = distance_loss_epoch_tuple[2]

            self.run_exp.log(
                wandb_dict,
            )
        return



    def print_training(self, episode_rewards, episode_ratio, episode_counter, ratio_recorder,
                       value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                       update_step, total_env_step, start_time, tag,
                       distance_loss_epoch_tuple=None,):

        if update_step % self.args.print_log_interval == 0 and len(episode_rewards) > 1:
            end_time = time()

            episodes_training_results = \
                "Train {} policy: {}\n" \
                "Updates {}, num timesteps {}, FPS {}, AvgTimePerUpdate: {:.3f}m\n" \
                "Last {} training episodes:\n" \
                "Mean/Median Reward {:.3f}/{:.3f}, Min/Max Reward {:.3f}/{:.3f}\n" \
                "Mean/Median Ratio {:.3f}/{:.3f}, Min/Max Ratio {:.3f}/{:.3f}\n" \
                "Mean/Median Counter {:.1f}/{:.1f}, Min/Max Counter {:.1f}/{:.1f}\n" \
                "The ratio threshold is {}\n" \
                "The value loss {:.5f}, the action loss {:.5f}, the entropy {:.5f}\n" \
                    .format(tag, self.experiment,
                            update_step, total_env_step,
                            int(total_env_step / (end_time - start_time)), (end_time - start_time) / update_step / 60,
                            len(episode_rewards),
                            np.mean(episode_rewards), np.median(episode_rewards),
                            np.min(episode_rewards), np.max(episode_rewards),
                            np.mean(episode_ratio), np.median(episode_ratio),
                            np.min(episode_ratio), np.max(episode_ratio),
                            np.mean(episode_counter), np.median(episode_counter),
                            np.min(episode_counter), np.max(episode_counter),
                            ratio_recorder,
                            value_loss_epoch, action_loss_epoch, dist_entropy_epoch,
                            )
            if distance_loss_epoch_tuple != None:
                episodes_training_results += (
                    "The distance loss {:.5f}, {:.5f}, {:.5f}\n".format(
                        distance_loss_epoch_tuple[0],
                        distance_loss_epoch_tuple[1],
                        distance_loss_epoch_tuple[2]
                    )
                )

            print(episodes_training_results)
        return
