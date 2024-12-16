import torch
import numpy as np
from numpy.random import randint
from torch.utils.data import Dataset
from utils.common import load_mat


class MultiviewDataset(Dataset):
    def __init__(self, root: str, device,
                 key_feature="data", key_label="labels",
                 normalize=True, views=None):
        """
        :param root: 数据集路径
        :param device: cpu or cuda
        :param key_feature: 存储特征的关键字
        :param key_label: 存储标签的关键字
        :param views: 指定读取哪些视图
        """
        # load_scene --- customize load_<dataset> function if not consistent
        data, labels = load_mat(root, views, key_feature=key_feature, key_label=key_label)
        num_view = len(data)
        view_dims = [0] * num_view
        for i in range(num_view):
            # .to(device)
            data[i] = torch.as_tensor(data[i], dtype=torch.float32).to(device)
            if normalize:
                max_value, _ = torch.max(data[i], dim=0, keepdim=True)
                min_value, _ = torch.min(data[i], dim=0, keepdim=True)
                data[i] = (data[i] - min_value) / (max_value - min_value + 1e-12)
                # data[i] = (data[i] - torch.min(data[i])) / (torch.max(data[i]) - torch.min(data[i]) + 1e-12)
            view_dims[i] = data[i].shape[1]
        self.data = data
        self.num_view = num_view
        self.view_dims = view_dims
        # .to(device)
        self.labels = torch.as_tensor(labels, dtype=torch.int32).view((-1,)).to(device)
        self.num_class = len(torch.unique(self.labels))
        self.device = device

    def __getitem__(self, index):
        item = []
        for i in range(self.num_view):
            # .to(self.device)
            item.append(self.data[i][index])
        # .to(self.device)
        return item, self.labels[index]

    def __len__(self):
        return len(self.labels)


class IncompleteMultiviewDataset(Dataset):
    def __init__(self, root: str, device, key_feature="data", key_label="labels", missing_rate=0.1,
                 normalize=True, views=None):
        """
        :param root:
        :param key_feature:
        :param key_label:
        :param missing_rate:
        """
        super(IncompleteMultiviewDataset, self).__init__()
        data, labels = load_mat(root, views, key_feature=key_feature, key_label=key_label)
        num_smp = len(labels)
        num_view = len(data)
        # create indicate matrix according to missing_rate
        # shape: num_smp * num_view
        # element: {0, 1} 1 if instance_i_v exists else 0
        indicate_matrix = torch.as_tensor(get_mask(num_view, num_smp, missing_rate))
        view_dims = [0] * num_view
        for i in range(num_view):
            data[i] = torch.as_tensor(data[i], dtype=torch.float32).to(device)
            if normalize:
                index_exist = torch.where(indicate_matrix[:, i] > 0)
                max_value, _ = torch.max(data[i][index_exist], dim=0, keepdim=True)
                min_value, _ = torch.min(data[i][index_exist], dim=0, keepdim=True)
                data[i] = (data[i] - min_value) / (max_value - min_value + 1e-12)
            view_dims[i] = data[i].shape[1]
        self.indicate_matrix = indicate_matrix
        self.data = data
        self.view_dims = view_dims
        self.num_view = num_view
        self.num_class = len(torch.unique(self.labels))
        self.labels = torch.as_tensor(labels, dtype=torch.int32).view((-1,)).to(device)

    def __getitem__(self, index):
        item = []
        for i in range(self.num_view):
            item.append(torch.as_tensor(self.data[i][index]))
        # 返回缺失情况
        return item, self.labels[index], self.indicate_matrix[index]

    def __len__(self):
        return len(self.labels)


def get_mask(view_num, data_len, missing_rate):
    mask = np.ones((data_len, view_num), dtype=np.float32)
    keep_rate = 1 - missing_rate
    keep_len = int(data_len * keep_rate)
    rand_perm = np.random.permutation(data_len)
    # 对于剩下的data_len - keep_len个
    rand_keep_view = np.random.randint(1, view_num, size=(data_len - keep_len,))
    # 对于剩下的每一行
    for i in range(data_len - keep_len):
        keep_view = np.zeros((view_num,))
        index_to_keep = np.random.choice(np.arange(0, view_num), rand_keep_view[i], replace=False)
        keep_view[index_to_keep] = 1
        mask[rand_perm[keep_len + i]] *= keep_view
    rand_perm = np.random.permutation(data_len)
    return mask[rand_perm]


