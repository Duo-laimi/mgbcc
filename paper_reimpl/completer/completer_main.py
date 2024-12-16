import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os.path as path
import argparse
from utils.common import load_json, init_torch
from utils.dataset import MultiviewDataset
from utils.score import cluster_metric
from completer_model.model import Completer
from sklearn.cluster import KMeans


def main():
    # 读取命令行参数
    parser = argparse.ArgumentParser(description="Command Line Params")
    parser.add_argument("--dataset", type=str, default="ALOI_100", help="Dataset used for Training.")
    parser.add_argument("--global_config", type=str, default="../../config/completer_config/global.json", help="The path of global config files.")
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
    mv_dataset = MultiviewDataset(ds_path, global_params["device"], views=ds_config["select_views"], normalize=ds_config["normalize"])
    # 构建数据加载器
    batch_size = ds_config["batch_size"]
    dataloader = DataLoader(mv_dataset, batch_size, shuffle=True, num_workers=0)
    # 构建模型
    completer = Completer(mv_dataset.view_dims,
                          ds_config["latent_dim"],
                          ds_config["autoencoder"]["mid_archs"],
                          ds_config["prediction"]["mid_archs"],
                          ds_config["alpha"],
                          ds_config["use_linear_projection"])

    completer.to(global_params["device"])
    summary(completer.mv_aes[0], (mv_dataset.view_dims[0],))
    summary(completer.mv_pres[0], (ds_config["latent_dim"],))
    # 优化器
    optimizer = Adam(completer.parameters(),
                                 lr=global_params["learning_rate"],
                                 weight_decay=global_params["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=global_params["epochs"], eta_min=0.)

    weights = ds_config["loss_weights"]
    kmeans = KMeans(n_clusters=mv_dataset.num_class, n_init="auto", random_state=global_params["seed"])
    for epoch in range(global_params["epochs"]):
        completer.train()
        for bid, (x, y) in enumerate(dataloader):
            loss_con, loss_rec, loss_cp, _ = completer(x)
            loss = loss_con * weights[0] + loss_rec * weights[1]
            if epoch+1 > global_params["start_dual_prediction"]:
                loss += loss_cp * weights[2]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        completer.eval()
        with torch.no_grad():
            loss_con, loss_rec, loss_cp, h = completer(mv_dataset.data)
            # 拼接不同视图的特征
            h = torch.concat(h, dim=1).detach().cpu().numpy()
            # k_means
            y_pred = kmeans.fit_predict(h)
            acc, nmi, pur = cluster_metric(mv_dataset.labels.cpu().numpy(), y_pred)
            print(f"epoch {epoch+1}, loss_con {loss_con}, loss_rec: {loss_rec}, loss_cp: {loss_cp}, "
                  f"acc {acc}, nmi {nmi}, pur {pur}")


if __name__ == "__main__":
    main()







