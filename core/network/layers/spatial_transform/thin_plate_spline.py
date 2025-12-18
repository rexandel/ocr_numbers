import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class thin_plate_spline(nn.Module):
    def __init__(self, fiducial_points, input_channels, height, width):
        super(thin_plate_spline, self).__init__()
        self.f = fiducial_points
        self.ch = input_channels
        self.h = height
        self.w = width
        self.loc_net = _localization_net(self.f, self.ch)
        self.grid_gen = _grid_generator(self.f, self.h, self.w)

    def forward(self, x):
        ctrl_pts = self.loc_net(x)
        sampling_grid = self.grid_gen.compute_grid(ctrl_pts)
        sampling_grid = sampling_grid.reshape([ctrl_pts.size(0), self.h, self.w, 2])
        return F.grid_sample(x, sampling_grid, padding_mode='border', align_corners=True)


class _localization_net(nn.Module):
    def __init__(self, fiducial_points, input_channels):
        super(_localization_net, self).__init__()
        self.f = fiducial_points
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.fc2 = nn.Linear(256, self.f * 2)
        self.fc2.weight.data.fill_(0)
        ctrl_x = np.linspace(-1.0, 1.0, int(fiducial_points / 2))
        ctrl_y_top = np.linspace(0.0, -1.0, num=int(fiducial_points / 2))
        ctrl_y_bot = np.linspace(1.0, 0.0, num=int(fiducial_points / 2))
        pts_top = np.stack([ctrl_x, ctrl_y_top], axis=1)
        pts_bot = np.stack([ctrl_x, ctrl_y_bot], axis=1)
        init_bias = np.concatenate([pts_top, pts_bot], axis=0)
        self.fc2.bias.data = torch.from_numpy(init_bias).float().view(-1)

    def forward(self, x):
        bs = x.size(0)
        feat = self.conv_layers(x).view(bs, -1)
        return self.fc2(self.fc1(feat)).view(bs, self.f, 2)


class _grid_generator(nn.Module):
    def __init__(self, fiducial_points, height, width):
        super(_grid_generator, self).__init__()
        self.eps = 1e-6
        self.f = fiducial_points
        self.h = height
        self.w = width
        self.ctrl = self._init_ctrl_points()
        self.target = self._init_target_points()
        self.register_buffer("inv_delta", torch.tensor(self._compute_inv_delta()).float())
        self.register_buffer("p_hat", torch.tensor(self._compute_p_hat()).float())

    def _init_ctrl_points(self):
        x = np.linspace(-1.0, 1.0, int(self.f / 2))
        y_top = -1 * np.ones(int(self.f / 2))
        y_bot = np.ones(int(self.f / 2))
        top = np.stack([x, y_top], axis=1)
        bot = np.stack([x, y_bot], axis=1)
        return np.concatenate([top, bot], axis=0)

    def _compute_inv_delta(self):
        hat = np.zeros((self.f, self.f), dtype=float)
        for i in range(self.f):
            for j in range(i, self.f):
                r = np.linalg.norm(self.ctrl[i] - self.ctrl[j])
                hat[i, j] = r
                hat[j, i] = r
        np.fill_diagonal(hat, 1)
        hat = (hat ** 2) * np.log(hat)
        delta = np.concatenate([
            np.concatenate([np.ones((self.f, 1)), self.ctrl, hat], axis=1),
            np.concatenate([np.zeros((2, 3)), np.transpose(self.ctrl)], axis=1),
            np.concatenate([np.zeros((1, 3)), np.ones((1, self.f))], axis=1)
        ], axis=0)
        return np.linalg.inv(delta)

    def _init_target_points(self):
        grid_x = (np.arange(-self.w, self.w, 2) + 1.0) / self.w
        grid_y = (np.arange(-self.h, self.h, 2) + 1.0) / self.h
        p = np.stack(np.meshgrid(grid_x, grid_y), axis=2)
        return p.reshape([-1, 2])

    def _compute_p_hat(self):
        n = self.target.shape[0]
        p_tile = np.tile(np.expand_dims(self.target, axis=1), (1, self.f, 1))
        c_tile = np.expand_dims(self.ctrl, axis=0)
        diff = p_tile - c_tile
        rbf_norm = np.linalg.norm(diff, ord=2, axis=2, keepdims=False)
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
        return np.concatenate([np.ones((n, 1)), self.target, rbf], axis=1)

    def compute_grid(self, ctrl_pts):
        bs = ctrl_pts.size(0)
        device = ctrl_pts.device
        inv_delta_batch = self.inv_delta.repeat(bs, 1, 1).to(device)
        p_hat_batch = self.p_hat.repeat(bs, 1, 1).to(device)
        pts_with_zeros = torch.cat((ctrl_pts, torch.zeros(bs, 3, 2, device=device)), dim=1)
        t = torch.bmm(inv_delta_batch, pts_with_zeros)
        return torch.bmm(p_hat_batch, t)

