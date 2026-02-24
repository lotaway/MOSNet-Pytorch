#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSame(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, bias=True)

    def forward(self, x):
        ih, iw = x.shape[-2], x.shape[-1]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh = int(np.ceil(ih / sh))
        ow = int(np.ceil(iw / sw))
        pad_h = max((oh - 1) * sh + kh - ih, 0)
        pad_w = max((ow - 1) * sw + kw - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return self.conv(x)


class _ConvFrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [1, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128]
        strides = [(1, 1), (1, 1), (1, 3)] * 4
        self.convs = nn.ModuleList([
            Conv2dSame(channels[i], channels[i + 1], kernel_size=(3, 3), stride=strides[i])
            for i in range(12)
        ])

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        for conv in self.convs:
            x = F.relu(conv(x))
        x = x.permute(0, 2, 1, 3).contiguous()
        b, t, c, f = x.shape
        return x.view(b, t, c * f)


class TorchCNNBLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = _ConvFrontEnd()
        self.blstm = nn.LSTM(512, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(256, 128)
        self.frame = nn.Linear(128, 1)

    def forward(self, x):
        x = self.frontend(x)
        if x.shape[-1] != 512:
            raise RuntimeError(f"Expected BLSTM input dim 512, got {x.shape[-1]}")
        x, _ = self.blstm(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        frame = self.frame(x)
        avg = frame.mean(dim=1)
        return avg, frame


class TorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = _ConvFrontEnd()
        self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(512, 64)
        self.frame = nn.Linear(64, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        frame = self.frame(x)
        avg = frame.mean(dim=1)
        return avg, frame


class TorchBLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.blstm = nn.LSTM(257, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(256, 64)
        self.frame = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.blstm(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        frame = self.frame(x)
        avg = frame.mean(dim=1)
        return avg, frame


def build_model(model_name: str) -> nn.Module:
    if model_name == "CNN":
        return TorchCNN()
    if model_name == "BLSTM":
        return TorchBLSTM()
    if model_name == "CNN-BLSTM":
        return TorchCNNBLSTM()
    raise ValueError("please specify model to train with, CNN, BLSTM or CNN-BLSTM")
