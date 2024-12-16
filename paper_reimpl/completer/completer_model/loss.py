import torch
import torch.nn as nn


# Cross-view Contrastive Loss
def CrossViewContrastiveLoss(p0, p1, alpha=9):
    """
        :param p0: 视图0的概率分配
        :param p1: 视图1的概率分配
        :return: 损失函数值
    """
    # 分别计算联合分布和边缘分布
    num_smp = p0.shape[0]
    # N x D -> D x N
    p0, p1 = p0.T, p1.T
    p_joint = p0 @ p1.T / num_smp
    p_joint = (p_joint + p_joint.T) / 2
    p_joint = p_joint / p_joint.sum()
    # 计算边缘分布
    p0_margin = p_joint.sum(1, keepdim=True)
    p1_margin = p_joint.sum(0, keepdim=True)
    p_tmp = p0_margin @ p1_margin
    loss = - p_joint * (torch.log(p_joint) - (alpha+1) * torch.log(p_tmp))
    return loss.sum()


