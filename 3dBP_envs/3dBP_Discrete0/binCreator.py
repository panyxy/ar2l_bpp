import numpy as np
import copy
import torch

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box(self, **kwargs):
        pass

    def preview(self, box_num=1):
        while len(self.box_list) < box_num:
            self.generate_box()
        return copy.deepcopy(self.box_list[:box_num])

    def drop_box(self, index=0):
        assert len(self.box_list) >= 0
        self.box_list.pop(index)


class RandomSeqCreator(BoxCreator):
    default_box_set = []
    default_max_size = {'width': 5, 'length': 5, 'height': 5}
    for i in range(default_max_size['width']):
        for j in range(default_max_size['length']):
            for k in range(default_max_size['height']):
                default_box_set.append((2+i, 2+j, 2+k))

    def __init__(self, box_set=None, setting=1):
        super(RandomSeqCreator, self).__init__()
        self.box_set = box_set
        if self.box_set is None:
            self.box_set = RandomSeqCreator.default_box_set

        if setting == 3:
            self.box_density = lambda: -np.random.random((150, 1)) + 1
        else:
            self.box_density = lambda: np.ones((150, 1), dtype=np.float)

    def reset(self):
        self.box_list.clear()
        idx = np.random.choice(np.arange(len(self.box_set)), size=150, replace=True)
        self.boxes = np.concatenate((np.array(self.box_set)[idx, ...], self.box_density()), axis=-1)
        self.box_idx = 0

    def generate_box(self, **kwargs):
        self.box_list.append(self.boxes[self.box_idx])
        self.box_idx += 1



class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):
        super(LoadBoxCreator, self).__init__()
        self.data_name = data_name
        self.traj_index = 0
        self.box_index = 0
        self.box_trajs = torch.load(self.data_name)
        self.traj_nums = len(self.box_trajs)

    def reset(self, traj_index=None):
        self.box_list.clear()
        self.recorder = []
        self.traj_index = self.traj_index % self.traj_nums if traj_index is None else traj_index
        self.boxes = np.array(self.box_trajs[self.traj_index]).tolist()
        self.traj_index += 1
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([100, 100, 100])

    def generate_box(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1