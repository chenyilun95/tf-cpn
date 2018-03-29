### TODO(still not work out the most humane way)

import os
import os.path as osp
import sys
sys.setrecursionlimit(10000)

class ConfigBase(object):

    def __init__(self, username, lr, optimizer):
        self.username = username

    display = 1

    gpu_ids = '0'
    nr_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.nr_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using /gpu:{}'.format(self.gpu_ids))
