import copy
import sys

import numpy as np
import gym
import torch
import random

try:
    from .space import Space
    from .binCreator import BoxCreator, LoadBoxCreator, RandomSeqCreator
except:
    from space import Space
    from binCreator import BoxCreator, LoadBoxCreator, RandomSeqCreator
from utils import get_args, registration_envs


class PackingContinuous(gym.Env):
    def __init__(self,
                 setting=1,
                 container_size=(10, 10, 10),
                 item_set=None,
                 data_name=None,
                 load_test_data=False,
                 num_box = 40,
                 num_next_box = 1,
                 num_candidate_position = 120,
                 node_dim=9,
                 sample_from_distribution=True,
                 sample_left_bound=0.1,
                 sample_right_bound=0.5,
                 unit_interval=0.1,
                 sparse_reward=True,
                 num_env=1,
                 env_idx=0,
                 num_load_episodes=100,
                 **kwargs,
                 ):

        """
        bin_state:   x, y, z, w, l, h, density, 0, isEmbed
        box_state:   0, 0, 0, w, l, h, density, 0, isEmbed
        constraint:  x, y, z, w, l, h, density, isFeasi, isEmbed
        """
        super(PackingContinuous, self).__init__()

        self.num_box = num_box
        self.num_next_box = num_next_box
        self.node_dim = node_dim
        self.num_candidate_position = num_candidate_position

        self.bin_size = container_size
        self.setting = setting
        self.item_set = item_set
        self.orientation = 2 if setting != 2 else 6
        self.test = self.load_test_data = load_test_data
        self.sparse_reward = sparse_reward

        self.sample_from_distribution = sample_from_distribution
        if sample_from_distribution:
            self.sample_left_bound = sample_left_bound
            self.sample_right_bound = sample_right_bound
            self.unit_interval = unit_interval

            self.minimum_size = sample_left_bound
            self.maximum_size = sample_right_bound
        else:
            self.minimum_size = np.min(np.array(item_set))
            self.maximum_size = np.max(np.array(item_set))

        self.num_env = num_env
        self.env_idx = env_idx
        self.num_load_episodes = num_load_episodes

        self.space = Space(*self.bin_size, self.minimum_size, self.maximum_size, self.num_box)

        if not load_test_data:
            assert item_set is not None
            self.box_creator = RandomSeqCreator(
                item_set, setting, sample_from_distribution, sample_left_bound, sample_right_bound, unit_interval,
            )
        else:
            self.box_creator = LoadBoxCreator(data_name, num_env, env_idx, num_load_episodes)

        self.observation_size = (self.num_box + self.num_next_box + self.num_candidate_position) * self.node_dim
        self.observation_space = gym.spaces.Box(low=0., high=self.space.height, shape=((self.observation_size, )))
        self.action_space = gym.spaces.Discrete(n=self.num_candidate_position)

        self.next_box = None
        self.next_box_density = None

        self.packed_box = [(
            self.space.boxes[0].w, self.space.boxes[0].l, self.space.boxes[0].h,
            self.space.boxes[0].x, self.space.boxes[0].y, self.space.boxes[0].z,
            self.space.boxes[0].density, self.space.boxes[0].box_order,
        )]
        self.num_packed_box = len(self.packed_box)

        self.is_inner = False


    def reset(self):
        self.box_creator.reset()
        self.space.reset()

        self.next_box = None
        self.next_box_density = None

        self.packed_box = [(
            self.space.boxes[0].w, self.space.boxes[0].l, self.space.boxes[0].h,
            self.space.boxes[0].x, self.space.boxes[0].y, self.space.boxes[0].z,
            self.space.boxes[0].density, self.space.boxes[0].box_order,
        )]
        self.num_packed_box = len(self.packed_box)

        self.is_inner = False

        return self.perm_observation()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            self.SEED = seed
        return [seed]

    def get_box_ratio(self):
        assert self.next_box is not None
        return np.prod(np.array(self.next_box[0:3])) / np.prod(self.space.plain_size)

    def generate_next_box(self):
        return self.box_creator.preview(1)[0]

    def generate_next_n_box(self):
        return self.box_creator.preview(self.num_next_box)


    def generate_observed_bin(self, space):
        bin_nodes = copy.deepcopy(space.box_vec)
        graph_nodes = np.zeros((self.num_box, self.node_dim))
        bin_node_num = min(len(bin_nodes), self.num_box)
        graph_nodes[:bin_node_num, :] = np.array(bin_nodes)[-bin_node_num:]
        return graph_nodes


    def generate_observed_box(self, next_n_box, next_n_box_density):
        next_n_box_vec = np.zeros((self.num_next_box, self.node_dim))
        next_n_box_vec[:, 3:7] = np.concatenate((next_n_box, next_n_box_density), axis=-1)
        next_n_box_vec[:, -1] = 1.
        return next_n_box_vec


    def perm_observation(self):
        self.num_packed_box = len(self.packed_box)

        self.next_box = None
        self.next_box_density = None

        self.next_n_box = np.array(self.generate_next_n_box())
        if not self.test:
            self.next_n_box_density = self.next_n_box[:, 3:4]
            self.next_n_box = self.next_n_box[:, :3]
        else:
            if self.setting != 3:
                self.next_n_box_density = np.ones((self.next_n_box.shape[0], 1), dtype=np.float32)
            else:
                self.next_n_box_density = self.next_n_box[:, 3:4]
            self.next_n_box = self.next_n_box[:, :3]

        bin_configuration = self.generate_observed_bin(space=self.space)
        next_n_box_vec = self.generate_observed_box(self.next_n_box, self.next_n_box_density)
        observation = np.reshape(np.concatenate((bin_configuration, next_n_box_vec), axis=0), (-1,))

        return observation


    def pack_observation(self, box_idx):
        assert self.is_inner
        self.num_packed_box = len(self.packed_box)

        bin_configuration = self.generate_observed_bin(space=self.space)

        self.next_n_box, self.next_n_box_density = self.reorder_next_n_box(
            box_idx, self.next_n_box, self.next_n_box_density
        )
        next_n_box_vec = self.generate_observed_box(self.next_n_box, self.next_n_box_density)

        assert abs(np.prod(self.next_box) - np.prod(self.next_n_box[0])) < 1e-3
        assert abs(self.next_box_density - self.next_n_box_density[0]) < 1e-3

        feasible_position, infeasible_position = self.compute_feasible_ems(
            self.space, self.next_box, self.next_box_density, self.num_packed_box
        )

        candidate_position = self.postprocess_candidate_position(
            feasible_position, infeasible_position, self.next_box, self.next_box_density
        )
        observation = np.reshape(np.concatenate((bin_configuration, next_n_box_vec, candidate_position), axis=0), (-1,))
        return observation

    def reorder_next_n_box(self, box_idx, next_n_box, next_n_box_density):
        next_n_box = np.concatenate(
            (next_n_box[box_idx:box_idx + 1],
             next_n_box[0:box_idx],
             next_n_box[box_idx + 1:]), axis=0
        )
        next_n_box_density = np.concatenate(
            (next_n_box_density[box_idx:box_idx + 1],
             next_n_box_density[0:box_idx],
             next_n_box_density[box_idx + 1:]), axis=0
        )
        return next_n_box, next_n_box_density


    def postprocess_candidate_position(self, feasible_position, infeasible_position, next_box, next_box_density):
        if feasible_position.shape[0] >= self.num_candidate_position:
            candidate_position = feasible_position[:self.num_candidate_position, ...]
        else:
            candidate_position = np.concatenate((feasible_position, infeasible_position), axis=0)
            if candidate_position.shape[0] < self.num_candidate_position:
                padding = np.zeros((self.num_candidate_position - candidate_position.shape[0], *candidate_position.shape[1:]))
                padding[:, 3:7] = np.concatenate((next_box, [next_box_density]), axis=0)
                candidate_position = np.concatenate((candidate_position, padding), axis=0)
            candidate_position = candidate_position[:self.num_candidate_position, ...]
        return candidate_position

    def compute_feasible_ems(self, space, next_box, next_box_density, num_packed_box):
        possible_positions = space.EMSPoint(next_box, self.setting)

        feasible_position = np.zeros((0, self.node_dim), dtype=np.float)
        infeasible_position = np.zeros((0, self.node_dim), dtype=np.float)

        for position in possible_positions:
            x, y, z, x_w, y_l, z_h = position
            w, l, h = x_w - x, y_l - y, z_h - z

            isFeasible, feasible_z = space.drop_box_virtual(
                [w, l, h], (x, y), next_box_density, num_packed_box, self.setting
            )
            if isFeasible:
                feasible_position = np.concatenate(
                    (feasible_position, [[x, y, z, w, l, h, next_box_density, 1., 1.]]), axis=0
                )
            else:
                infeasible_position = np.concatenate(
                    (infeasible_position, [[x, y, z, w, l, h, next_box_density, 0., 1.]]), axis=0
                )
        self.clear_up_virtual_box(space)
        return feasible_position, infeasible_position

    def clear_up_virtual_box(self, space):
        for box in space.boxes:
            box.up_virtual_edges = dict()
        return

    def generate_ems(self, space, packed_box):
        space.GENEMS(
            [
                packed_box.x,
                packed_box.y,
                packed_box.z,
                round(packed_box.x + packed_box.w, 6),
                round(packed_box.y + packed_box.l, 6),
                round(packed_box.z + packed_box.h, 6),
            ]
        )
        return



    def decode_action(self, action):
        x, y, z, w, l, h, density, box_idx, self.is_inner = action
        box_idx = int(box_idx)

        if not self.is_inner:
            assert abs(w * l * h - np.prod(self.next_box)) < 1e-3 and abs(self.next_box_density - density) < 1e-3
            assert abs(w * l * h - np.prod(self.next_n_box[box_idx])) < 1e-3 and abs(self.next_n_box_density[box_idx][0] - density) < 1e-3
        else:
            self.next_box = self.next_n_box[box_idx]
            self.next_box_density = self.next_n_box_density[box_idx][0]

            self.box_creator.move_box(box_idx)
            assert abs(w * l * h - np.prod(self.next_box)) < 1e-3 and abs(self.next_box_density - density) < 1e-3

        return x, y, z, w, l, h, density, box_idx


    def step(self, action):
        x, y, z, w, l, h, density, box_idx = self.decode_action(action)

        if not self.is_inner:
            succeeded = self.space.drop_box([w, l, h], [x,y], self.next_box_density, self.setting)

            if not succeeded:
                ratio = self.space.get_ratio()
                reward = ratio * 10 * self.sparse_reward
                done = True
                info = {'counter': len(self.space.boxes) - 1,
                        'ratio': ratio,
                        'reward': reward,
                        'packed_box': self.packed_box[1:],
                        }
                return self.perm_observation(), reward, done, info

            self.packed_box.append(
                (
                    self.space.boxes[-1].w, self.space.boxes[-1].l, self.space.boxes[-1].h,
                    self.space.boxes[-1].x, self.space.boxes[-1].y, self.space.boxes[-1].z,
                    self.space.boxes[-1].density, self.space.boxes[-1].box_order,
                )
            )
            box_ratio = self.get_box_ratio()
            reward =  box_ratio * 10 * (1 - self.sparse_reward)
            done = False
            info = {'counter': len(self.space.boxes)-1}

            self.generate_ems(self.space, self.space.boxes[-1])
            self.box_creator.drop_box(index=0)

            return self.perm_observation(), reward, done, info

        else:
            return self.pack_observation(box_idx), 0., False, {}



