import torch
import numpy as np
import warnings
from sklearn.cluster import KMeans, k_means
from sklearn.exceptions import ConvergenceWarning


# For Clustering
class GranularBall:
    """class of the granular ball"""

    def __init__(self, data, labels, indices):
        """
        :param data: 样本特征
        :param labels: 样本标签
        :param indices: 样本索引
        """
        self.data = data
        self.labels = labels
        self.indices = np.array([indices]).squeeze().reshape(-1,)
        self.num_smp, self.dim = data.shape
        self.center = self.data.mean(0)
        arr = torch.norm(self.data - self.center, p=2, dim=1)
        # 各点到中心点距离的平均值
        self.r = arr.mean()

    def split_balls(self, p):
        # 将该粒球划分为k个小的粒球
        k = max(self.num_smp // p, 1)
        data = self.data.detach().cpu().numpy()
        # 当data中有大量重复时
        k = min(k, np.unique(data, axis=0).shape[0])
        if p == 1:
            y_part = np.arange(data.shape[0])
        else:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            y_part = kmeans.fit_predict(data)
        # 根据标签对样本进行划分
        y_part = torch.from_numpy(y_part)
        sub_balls = []
        for i in range(k):
            indices = torch.where(y_part == i)
            sub_data = self.data[indices]
            sub_labels = self.labels[indices]
            sub_indices = self.indices[indices]
            new_ball = GranularBall(sub_data, sub_labels, sub_indices)
            sub_balls.append(new_ball)
        return sub_balls


class GBList:
    """class of the list of granular ball"""

    def __init__(self, data, labels, p=8):
        self.data = data
        self.labels = labels
        self.indices = np.arange(data.shape[0])
        self.granular_balls = [GranularBall(data, labels, self.indices)]  # gbs is initialized with all data
        self.split_granular_balls(p)

    def __len__(self):
        return len(self.granular_balls)

    def __getitem__(self, i):
        return self.granular_balls[i]

    def split_granular_balls(self, p):
        """
        Split the balls, initialize the balls list.
        :param p: If the number of samples of a ball is less than this value, stop splitting.
        """
        gb_list = []
        for ball in self:
            gb_list.extend(ball.split_balls(p))
        self.granular_balls = gb_list

    def get_centers(self):
        """
        :return: the center of each ball.
        """
        return torch.vstack(list(map(lambda x: x.center, self.granular_balls)))

    def get_rs(self):
        """
        :return: 返回半径r
        """
        return torch.vstack(list(map(lambda x: x.r, self.granular_balls))).squeeze()

    def get_data(self):
        """
        :return: Data from all existing granular balls in the GBlist.
        """
        list_data = [ball.data for ball in self.granular_balls]
        list_labels = [ball.labels for ball in self.granular_balls]
        list_indices = [ball.indices for ball in self.granular_balls]
        return torch.concat(list_data, dim=0), torch.concat(list_labels, dim=0), torch.concat(list_indices, dim=0)

    def del_ball(self, min_smp=0):
        T_ball = []
        for ball in self.granular_balls:
            if ball.num_smp >= min_smp:
                T_ball.append(ball)
        self.granular_balls = T_ball
        self.data, self.labels, self.indices = self.get_data()

    @torch.no_grad
    def affinity(self, spread=3):
        # 获取所有中心点
        centers = self.get_centers()
        # 计算不同中心点的距离
        dist = torch.cdist(centers, centers)
        # 获取所有半径
        rs = self.get_rs()
        extra = rs.unsqueeze(0) + rs.unsqueeze(-1)
        indicate = dist <= extra
        indicate = torch.where(indicate, 1, 0).type(torch.float32)
        # indicate = transitive_neighbor_relations(indicate, spread)
        return indicate


class MVGBList:
    def __init__(self, mv_data, labels, p=8):
        """
        :param mv_data: 多视图数据
        :param labels:
        :param p:
        """
        self.num_view = len(mv_data)
        self.gblists = []
        for i in range(self.num_view):
            gblist = GBList(mv_data[i], labels, p=p)
            self.gblists.append(gblist)

    def __len__(self):
        return self.num_view

    def __getitem__(self, i):
        return self.gblists[i]


# 是否包含同一个样本，针对不同视图的两个粒球
def contain_same_sample(ball0: GranularBall, ball1: GranularBall):
    n0, n1 = ball0.num_smp, ball1.num_smp
    for i in range(n0):
        for j in range(n1):
            if ball0.indices[i] == ball1.indices[j]:
                return True
    return False


# 传递邻接（重叠）关系
def transitive_neighbor_relations(a, k=3):
    while k > 0:
        a_ = torch.where(a @ a > 0, 1., 0.)
        a_ = torch.where(torch.logical_or(a, a_), 1., 0.)
        a = a_
        k -= 1
    return a
