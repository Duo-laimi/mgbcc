import random
import numpy as np
import torch
import json
import hdf5storage as hdf
import itertools


def init_torch(seed):
    # 随机数种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_json(path,  encoding='utf-8'):
    with open(path, "r", encoding=encoding) as fp:
        params = json.load(fp)
    return params


def load_mat(path, views=None, key_feature="data", key_label="labels"):
    data = hdf.loadmat(path)
    feature = []
    num_view = len(data[key_feature])
    label = data[key_label].reshape((-1,))
    num_smp = label.size
    for v in range(num_view):
        tmp = data[key_feature][v][0].squeeze()
        feature.append(tmp)
    # 打乱样本
    rand_permute = np.random.permutation(num_smp)
    for v in range(num_view):
        feature[v] = feature[v][rand_permute]
    label = label[rand_permute]
    if views is None or len(views) == 0:
        views = list(range(num_view))
    views_feature = [feature[v] for v in views]
    return views_feature, label


# 实现功能：返回参数组合的笛卡尔集
def get_all_parameters(*parameters_range):
    return itertools.product(*parameters_range)

