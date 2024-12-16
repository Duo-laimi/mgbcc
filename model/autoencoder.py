from typing import Sequence

import torch
import torch.nn as nn
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feature_dim: int,
                 middle_dims: Sequence[int] = (1024, 512, 256),
                 use_linear_projection: bool = False):
        super(Encoder, self).__init__()
        middle_dims = [input_dim] + list(middle_dims) + [feature_dim]
        middle_layers = nn.ModuleList()
        for i in range(len(middle_dims) - 2):
            layer = [nn.Linear(middle_dims[i], middle_dims[i + 1])]
            if not use_linear_projection:
                layer.extend([
                    nn.BatchNorm1d(middle_dims[i + 1]),
                    nn.ReLU(inplace=True)
                ])
            middle_layers.append(nn.Sequential(*layer))
        middle_layers.append(nn.Linear(middle_dims[-2], middle_dims[-1]))
        # Completer在编码层加了一层Softmax
        # middle_layers.append(nn.Softmax(dim=1))
        self.middle_layers = middle_layers
        self.middle_dims = middle_dims

    def forward(self, x):
        for layer in self.middle_layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 feature_dim: int,
                 output_dim: int,
                 middle_dims: Sequence[int] = (256, 512, 1024),
                 use_linear_projection: bool = False):
        super(Decoder, self).__init__()
        middle_dims = [feature_dim] + list(middle_dims) + [output_dim]
        middle_layers = nn.ModuleList()
        for i in range(len(middle_dims) - 2):
            layer = [nn.Linear(middle_dims[i], middle_dims[i + 1])]
            if not use_linear_projection:
                layer.extend([
                    nn.BatchNorm1d(middle_dims[i + 1]),
                    nn.ReLU(inplace=True)
                ])
            middle_layers.append(nn.Sequential(*layer))
        middle_layers.append(nn.Linear(middle_dims[-2], middle_dims[-1]))
        # 控制输出范围在0-1之间
        middle_layers.append(nn.Sigmoid())
        self.middle_layers = middle_layers
        self.middle_dims = middle_dims

    def forward(self, x):
        for layer in self.middle_layers:
            x = layer(x)
        return x


# 重构和Prediction都可以使用自编码器的结构
class AutoEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feature_dim: int,
                 middle_dims: Sequence[int] = (1024, 512, 256),
                 use_linear_projection: bool = False):
        super(AutoEncoder, self).__init__()
        output_dim = input_dim
        self.middle_dims = [input_dim] + list(middle_dims) + [feature_dim]
        self.encoder = Encoder(input_dim, feature_dim, middle_dims, use_linear_projection)
        self.decoder = Decoder(feature_dim, output_dim, middle_dims[::-1], use_linear_projection)

    def forward(self, x):
        hidden = self.encoder(x)
        x_rec = self.decoder(hidden)
        return hidden, x_rec

    def pretrain(self, x, epochs=100, lr=1e-2, weight_decay=0.):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            h, x_rs = self(x)
            loss = criterion(x, x_rs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 返回最后一次的结果
        return h, x_rs


# 仅编码器
class MultiviewPrediction(nn.Module):
    def __init__(self, view_dims, latent_dim, middle_encoders, use_linear_projection=False):
        super(MultiviewPrediction, self).__init__()
        self.num_view = len(view_dims)
        # 构建num_view个编码器
        encoder_list = nn.ModuleList()
        for i in range(self.num_view):
            if middle_encoders is None:
                encoder = Encoder(view_dims[i], latent_dim, use_linear_projection=use_linear_projection)
            else:
                encoder = Encoder(view_dims[i], latent_dim, middle_encoders[i], use_linear_projection)
            encoder_list.append(encoder)
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        self.encoder_list = encoder_list

    def __getitem__(self, i):
        return self.encoder_list[i]

    def forward(self, x):
        hidden_list = []
        for i in range(self.num_view):
            encoder = self.encoder_list[i]
            hidden = encoder(x[i])
            hidden_list.append(hidden)
        return hidden_list


class MultiviewAutoEncoder(nn.Module):
    def __init__(self, view_dims, latent_dim, middle_encoders, use_linear_projection=False):
        super(MultiviewAutoEncoder, self).__init__()
        num_view = len(view_dims)
        # 构建num_view个自编码器
        autoencoder_list = nn.ModuleList()
        for i in range(num_view):
            if middle_encoders is None:
                autoencoder = AutoEncoder(view_dims[i], latent_dim, use_linear_projection=use_linear_projection)
            else:
                autoencoder = AutoEncoder(view_dims[i], latent_dim,
                                          middle_dims=middle_encoders[i], use_linear_projection=use_linear_projection)
            autoencoder_list.append(autoencoder)
        self.num_view = num_view
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        self.autoencoder_list = autoencoder_list

    def __getitem__(self, i):
        return self.autoencoder_list[i]

    def forward(self, x):
        """
        :param x: multi view data with shape num_view * [batch_size * feature_dim_v]
        :return: multi view latent feature with view-specific autoencoders, and reconstructed view
        """
        hidden_list, x_rs = [], []
        for i in range(self.num_view):
            autoencoder = self.autoencoder_list[i]
            hidden = autoencoder.encoder(x[i])
            hidden_list.append(hidden)
            x_r_view = autoencoder.decoder(hidden)
            x_rs.append(x_r_view)
        return hidden_list, x_rs

    def pretrain(self, dataloader, epochs, device, lr=1e-2, weight_decay=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
            for bid, (x, _) in loop:
                for i in range(len(x)):
                    x[i] = x[i].to(device)
                _, x_rs = self.forward(x)
                loss = torch.tensor(0., device=device)
                for i in range(len(x)):
                    loss += criterion(x[i], x_rs[i])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_description(desc=f"Pretrain Epoch [{epoch}/{epochs}]")
                loop.set_postfix(loss=loss.item())


class MultiviewAutoEncoderWithAvgpool(nn.Module):
    def __init__(self, view_dims, latent_dim, middle_encoders, use_linear_projection=False):
        super(MultiviewAutoEncoderWithAvgpool, self).__init__()
        num_view = len(view_dims)
        # 构建num_view个自编码器
        autoencoder_list = nn.ModuleList()
        for i in range(num_view):
            if middle_encoders is None:
                autoencoder = AutoEncoder(view_dims[i], latent_dim, use_linear_projection=use_linear_projection)
            else:
                autoencoder = AutoEncoder(view_dims[i], latent_dim,
                                          middle_dims=middle_encoders[i], use_linear_projection=use_linear_projection)
            autoencoder_list.append(autoencoder)
        self.num_view = num_view
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        self.autoencoder_list = autoencoder_list

    def forward(self, x):
        hidden_list = []
        for i in range(self.num_view):
            autoencoder = self.autoencoder_list[i]
            hidden = autoencoder.encoder(x[i])
            hidden_list.append(hidden)
        avg_hidden = torch.sum(torch.stack(hidden_list, dim=0), dim=0) / self.num_view
        x_r = []
        for i in range(self.num_view):
            autoencoder = self.autoencoder_list[i]
            x_r_view = autoencoder.decoder(avg_hidden)
            x_r.append(x_r_view)
        return avg_hidden, x_r, hidden_list


class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.normalize = nn.functional.normalize
        self.p = p
        self.dim = dim

    def forward(self, x):
        return self.normalize(x, p=self.p, dim=self.dim)
