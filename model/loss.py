import torch
import torch.nn as nn
from torch import Tensor

from typing import List, Tuple, Union


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        """
        :param temperature: 温度参数，调节平滑度
        """
        super(ContrastiveLoss, self).__init__()
        self.t = temperature

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
        neg_mask = torch.ones_like(pos_mask).to(device) - pos_mask
        idx = torch.arange(0, num_ins)
        # 修正正样本对掩码
        pos_mask[idx, idx] = 0
        # 缺失部分既不可以当做正样本，也不可以当做负样本
        # N * V -> N * V
        if indicate_matrix is None:
            indicate_matrix = torch.ones((num_smp, num_view), dtype=torch.float32).to(device)
        indicate_matrix_extend = indicate_matrix.view((-1, 1))
        base_mask = indicate_matrix_extend @ indicate_matrix_extend.T
        neg_mask = neg_mask * base_mask
        pos_mask = pos_mask * base_mask
        # 只要得到相应的掩码，就可以计算损失
        # 考虑用cross entropy 重写
        sim_pos = pos_mask * sim_x / self.t
        sim_neg = neg_mask * sim_x / self.t
        exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((num_ins, num_ins))
        expsum_sim = torch.exp(sim_pos) + exp_sim_neg
        # expsum_sim = exp_sim_neg
        loss = -(sim_pos - torch.log(expsum_sim) * pos_mask)
        avg_sim_pos = torch.sum(sim_pos) / torch.sum(pos_mask)
        avg_sim_neg = torch.sum(sim_neg) / (torch.sum(neg_mask))
        # return torch.sum(torch.as_tensor(loss)) / num_ins, avg_sim_pos, avg_sim_neg
        # 这里应该就是除以损失项的个数，所以应该是pos_mask.sum()
        return torch.sum(torch.as_tensor(loss)) / pos_mask.sum(), avg_sim_pos, avg_sim_neg


class ContrastiveLoss_v2(nn.Module):
    def __init__(self, temperature: float = 1.0):
        """
        :param temperature: 温度参数，调节平滑度
        :param neg_rate: 是否采样及相对于neg_rate的比例
        """
        super(ContrastiveLoss_v2, self).__init__()
        self.t = temperature

    def forward(self, x: Union[List[Tensor], Tuple[Tensor]], indicate_matrix=None):
        """
        :param x: 多视图数据，已经经过投影的特征，特征维度相同
        :param indicate_matrix:
        :return: 对比损失
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
        # 对比损失，这里要区分是否采样负样本，由于正样本比较少，因此总是全部使用
        pos_mask = torch.eye(num_smp).repeat((num_view, num_view)).to(device)
        neg_mask = torch.ones_like(pos_mask).to(device) - pos_mask
        idx = torch.arange(0, num_ins)
        # 修正正样本对掩码
        pos_mask[idx, idx] = 0
        logits = torch.exp(sim_x)
        log_prob = torch.log(logits) - torch.log((logits * neg_mask).sum(1, keepdim=True))
        mean_log_prob_pos = -(pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        loss = mean_log_prob_pos.mean()
        return loss, 0, 0
