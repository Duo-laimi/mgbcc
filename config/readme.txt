# 在命令行需要选择的参数
dataset
missing_rate

# 全局参数 global.json
src # 数据集的父路径
seed # 随机数种子
device=[cpu, cuda]


epochs # 训练轮次
batch_size # 批次大小
learning_rate # 学习率
weight_decay # 正则化

# 以下为单独每个数据集配置的参数 dataset.json
parent # 父级文件夹
dataset_name # 数据集的名称
# 完整路径为： [src]/[parent]/[dataset_name].mat
input_dims # 不同视图的输入特征的维度，列表
latent_dim # 所有视图投影特征的维度均相同
use_batch # 是否使用batch


use_linear_projection=[true, false] # 是否使用线性投影
projection_views # 不同视图的编码层网络结构，解码层对称

# 最多只有3个权重参数
# 这三个参数放在dataset.json中配置，但是尽量相同，减少更改
alpha0 # 损失权重0
alpha1 # 损失权重1
alpha2 # 损失权重2
