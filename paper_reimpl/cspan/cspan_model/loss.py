import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Union, Tuple
from model.loss import ContrastiveLoss


class InstanceAlignLoss(nn.Module):
    def __init__(self):
        """
        :param temperature: 温度参数，调节平滑度
        """
        super(InstanceAlignLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x: Union[List[Tensor], Tuple[Tensor]], indicate_matrix=None):
        """
        :param x: 多视图数据，已经经过投影的特征，特征维度相同
        :param indicate_matrix:
        :return: 对比损失, 正样本间的平均相似度，负样本间的平均相似度
        """
        num_view = len(x)
        num_smp = x[0].shape[0]
        num_ins = num_view * num_smp
        device = x[0].device
        # 沿第0个维度进行拼接
        x = torch.concat(x, dim=0)
        # 计算相似度，这里就是矩阵相乘
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
        pos_mask = torch.eye(num_smp).repeat((num_view, num_view)).to(device)
        idx = torch.arange(0, num_ins)
        # 修正正样本对掩码
        pos_mask[idx, idx] = 0
        # 缺失部分既不可以当做正样本，也不可以当做负样本
        # N * V -> N * V
        if indicate_matrix is None:
            indicate_matrix = torch.ones((num_smp, num_view), dtype=torch.float32).to(device)
        indicate_matrix_extend = indicate_matrix.view((-1, 1))
        base_mask = indicate_matrix_extend @ indicate_matrix_extend.T
        pos_mask = pos_mask * base_mask
        sim_pos = sim_x * pos_mask
        loss = self.criterion(sim_pos, pos_mask)
        return loss


class PrototypeAlignLoss(nn.Module):
    def __init__(self, no_negative=True):
        super(PrototypeAlignLoss, self).__init__()
        self.criterion = InstanceAlignLoss() if no_negative else ContrastiveLoss()
        self.no_negative = no_negative

    def forward(self, x: Union[List[Tensor], Tuple[Tensor]]):
        centers = []
        # 按道理来说应该先找到不同原型的对应关系
        for v in range(len(x)):
            centers.append(x[v].T)
        res = self.criterion(centers, None)
        return res if self.no_negative else res[0]


class PrototypeAlignLoss_v2(nn.Module):
    def __init__(self, no_negative=True):
        super(PrototypeAlignLoss_v2, self).__init__()

    def forward(self, x: Union[List[Tensor], Tuple[Tensor]]):
        centers = []
        #
        pass

