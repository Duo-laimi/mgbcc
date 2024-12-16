import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os.path as path
import argparse
from model.autoencoder import MultiviewAutoEncoder, Normalize

from cspan_model.loss import InstanceAlignLoss, PrototypeAlignLoss

from utils.common import load_json, init_torch
from utils.dataset import MultiviewDataset
from utils.score import cluster_metric
from sklearn.cluster import KMeans


def main():
    # 读取命令行参数
    parser = argparse.ArgumentParser(description="Command Line Params")
    parser.add_argument("--dataset", type=str, default="Caltech101-20", help="Dataset used for Training.")
    parser.add_argument("--global_config", type=str, default="../../config/cspan_config/global.json", help="The path of global config files.")
    args = parser.parse_args()
    # 读取全局参数
    global_params = load_json(args.global_config)
    # 设置数据集相关设置的路径，默认与全局配置在同一文件夹下
    if global_params["config"] is None:
        config_path = path.dirname(args.global_config)
        global_params["config"] = path.join(config_path, args.dataset + ".json")
    # 随机数种子
    init_torch(seed=global_params["seed"])
    # 读取数据集特定的参数
    ds_config = load_json(global_params["config"])
    # 读取数据集
    src = global_params["src"]
    ds_path = path.join(src, ds_config["parent"], args.dataset + ".mat")
    mv_dataset = MultiviewDataset(ds_path, global_params["device"], views=ds_config["select_views"])
    # 构建数据加载器
    if ds_config["use_batch"]:
        batch_size = global_params["batch_size"]
    else:
        batch_size = len(mv_dataset)
    dataloader = DataLoader(mv_dataset, batch_size, shuffle=True, num_workers=0)
    # 构建模型
    mv_aes = MultiviewAutoEncoder(mv_dataset.view_dims, ds_config["latent_dim"],
                                  ds_config["autoencoder"]["mid_archs"], ds_config["use_linear_projection"])
    # 在编码层后，加一层标准化层
    for v in range(mv_dataset.num_view):
        mv_aes[v].encoder.middle_layers.append(Normalize())
    mv_aes.to(global_params["device"])
    summary(mv_aes[0], (mv_dataset.view_dims[0],))
    # 优化器
    optimizer = Adam(mv_aes.parameters(),
                                 lr=global_params["learning_rate"],
                                 weight_decay=global_params["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=global_params["epochs"], eta_min=0.)
    weights = ds_config["loss_weights"]
    kmeans = KMeans(n_clusters=mv_dataset.num_class, n_init="auto", random_state=global_params["seed"])
    criterion_rec = nn.MSELoss()
    criterion_ins = InstanceAlignLoss()
    criterion_pro = PrototypeAlignLoss(False)
    for epoch in range(global_params["epochs"]):
        mv_aes.train()
        for bid, (x, y) in enumerate(dataloader):
            hs, x_rs = mv_aes(x)
            # 重建损失
            loss_rec = torch.tensor(0., device=global_params["device"])
            for v in range(mv_dataset.num_view):
                loss_rec += criterion_rec(x[v], x_rs[v])
            loss_ins = criterion_ins(hs)
            loss_pro = criterion_pro(hs)
            loss = loss_rec * weights[0] + loss_ins * weights[1] + loss_pro * weights[2]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        mv_aes.eval()
        with torch.no_grad():
            hs, _ = mv_aes(mv_dataset.data)
            # 拼接不同视图的特征
            # hs = torch.concat(hs, dim=1).detach().cpu().numpy()
            # hs = torch.stack(hs, dim=0).mean(0).detach().cpu().numpy()
            hs = torch.stack(hs, dim=0).mean(0)
            # k_means
            y_pred = kmeans.fit_predict(hs)
            acc, nmi, pur = cluster_metric(mv_dataset.labels.cpu().numpy(), y_pred)
            print(f"epoch {epoch+1}, loss_rec {loss_rec}, loss_ins: {loss_ins}, loss_pro: {loss_pro} "
                  f"acc {acc}, nmi {nmi}, pur {pur}")


if __name__ == "__main__":
    main()







