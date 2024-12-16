import torch
import torch.nn as nn
from model.autoencoder import MultiviewAutoEncoder, MultiviewPrediction
from .loss import CrossViewContrastiveLoss


class Completer(nn.Module):
    def __init__(self, view_dims, latent_dim, mid_archs_ae, mid_archs_pre, alpha=9, use_linear_projection=False):
        """
        需要配置的参数有：输入的特征维度，中间层， 跨视图重建层的结构
        :param view_dims: 输入特征的维度
        :param latent_dim: 投影特征的维度
        :mid_archs_ae: 自编码器的中间层
        :mid_archs_pre: 跨视图重建的中间层
        """
        super(Completer, self).__init__()
        self.num_view = len(view_dims)
        mv_aes = MultiviewAutoEncoder(view_dims, latent_dim, mid_archs_ae, use_linear_projection=use_linear_projection)
        mv_pres = MultiviewPrediction([latent_dim] * self.num_view, latent_dim, mid_archs_pre, use_linear_projection=use_linear_projection)

        # 在mv_aes的编码层加一层softmax
        for v in range(self.num_view):
            mv_aes[v].encoder.middle_layers.append(nn.Softmax(dim=1))
            mv_pres[v].middle_layers.append(nn.Softmax(dim=1))

        self.view_dims = view_dims
        self.latent_dim = latent_dim
        self.mv_aes = mv_aes
        self.mv_pres = mv_pres
        self.criterion_rec = nn.MSELoss()
        self.criterion_con = CrossViewContrastiveLoss
        self.alpha = alpha

    def forward(self, x):
        """
        :param x: 多视图数据
        :return: 损失函数
        """
        # 重建损失
        h, x_rs = self.mv_aes(x)
        loss_rec, loss_cp, loss_con = torch.zeros(3, dtype=torch.float32, device=x[0].device)
        for v in range(self.num_view):
            loss_rec = loss_rec + self.criterion_rec(x[v], x_rs[v])
        # 跨视图重建 如果要拓展到多个视图，则采用循环重建的想法 0->1 1->2 2->0
        h_cp = self.mv_pres(h)
        for v in range(self.num_view):
            loss_cp = loss_cp + self.criterion_rec(h_cp[v], h[(v + 1) % self.num_view])
        # loss_cp /= self.num_view
        # 对比学习, 这里采用两两对比
        for v0 in range(self.num_view - 1):
            for v1 in range(v0 + 1, self.num_view):
                loss_con = loss_con + self.criterion_con(h[v0], h[v1], self.alpha)
        # loss_con /= self.num_view * (self.num_view-1) / 2
        return loss_con, loss_rec, loss_cp, h

