#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSame(nn.Module):
    """TensorFlow-style SAME padding for any stride."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        ih, iw = x.shape[-2], x.shape[-1]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh = int(np.ceil(ih / sh))
        ow = int(np.ceil(iw / sw))
        pad_h = max((oh - 1) * sh + kh - ih, 0)
        pad_w = max((ow - 1) * sw + kw - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            )
        return self.conv(x)


class TorchCNNBLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [1, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 128]
        strides = [(1, 1), (1, 1), (1, 3)] * 4
        self.convs = nn.ModuleList(
            [
                Conv2dSame(channels[i], channels[i + 1], kernel_size=(3, 3), stride=strides[i])
                for i in range(12)
            ]
        )
        self.blstm = nn.LSTM(
            input_size=4 * 128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.dense1 = nn.Linear(256, 128)
        self.frame = nn.Linear(128, 1)

    def forward(self, x):
        # Keras input: (B, T, 257). Torch conv expects (B, C, T, F).
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected 3D/4D input, got shape {tuple(x.shape)}")

        for conv in self.convs:
            x = F.relu(conv(x))

        # (B, C, T, F) -> (B, T, C*F)
        x = x.permute(0, 2, 1, 3).contiguous()
        b, t, c, f = x.shape
        x = x.view(b, t, c * f)
        if c * f != 512:
            raise RuntimeError(f"Expected BLSTM input dim 512, got {c * f}")

        x, _ = self.blstm(x)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        frame_score = self.frame(x)
        average_score = frame_score.mean(dim=1)
        return average_score, frame_score


def _set_conv(conv_module, kernel, bias):
    # Keras Conv2D: (kh, kw, in_c, out_c) -> Torch: (out_c, in_c, kh, kw)
    w = np.transpose(kernel, (3, 2, 0, 1))
    conv_module.conv.weight.data.copy_(torch.from_numpy(w))
    conv_module.conv.bias.data.copy_(torch.from_numpy(bias))


def _set_lstm_direction(lstm, direction, kernel, recurrent, bias):
    # Keras LSTM: kernel(in,4h), recurrent(h,4h), bias(4h)
    # Torch LSTM: weight_ih(4h,in), weight_hh(4h,h), bias_ih(4h), bias_hh(4h)
    suffix = "_reverse" if direction == "backward" else ""
    getattr(lstm, f"weight_ih_l0{suffix}").data.copy_(torch.from_numpy(kernel.T))
    getattr(lstm, f"weight_hh_l0{suffix}").data.copy_(torch.from_numpy(recurrent.T))
    getattr(lstm, f"bias_ih_l0{suffix}").data.copy_(torch.from_numpy(bias))
    getattr(lstm, f"bias_hh_l0{suffix}").data.zero_()


def load_keras_h5_weights(model: TorchCNNBLSTM, h5_path: str):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Please install h5py first: pip install h5py") from exc

    with h5py.File(h5_path, "r") as f:
        for i, conv in enumerate(model.convs, start=1):
            base = f"model_weights/conv2d_{i}/conv2d_{i}"
            _set_conv(conv, f[f"{base}/kernel:0"][()], f[f"{base}/bias:0"][()])

        lstm_base = "model_weights/bidirectional_1/bidirectional_1"
        _set_lstm_direction(
            model.blstm,
            "forward",
            f[f"{lstm_base}/forward_lstm_1/kernel:0"][()],
            f[f"{lstm_base}/forward_lstm_1/recurrent_kernel:0"][()],
            f[f"{lstm_base}/forward_lstm_1/bias:0"][()],
        )
        _set_lstm_direction(
            model.blstm,
            "backward",
            f[f"{lstm_base}/backward_lstm_1/kernel:0"][()],
            f[f"{lstm_base}/backward_lstm_1/recurrent_kernel:0"][()],
            f[f"{lstm_base}/backward_lstm_1/bias:0"][()],
        )

        dense_base = "model_weights/time_distributed_2/time_distributed_2"
        model.dense1.weight.data.copy_(torch.from_numpy(f[f"{dense_base}/kernel:0"][()].T))
        model.dense1.bias.data.copy_(torch.from_numpy(f[f"{dense_base}/bias:0"][()]))

        frame_base = "model_weights/Frame_score/Frame_score"
        model.frame.weight.data.copy_(torch.from_numpy(f[f"{frame_base}/kernel:0"][()].T))
        model.frame.bias.data.copy_(torch.from_numpy(f[f"{frame_base}/bias:0"][()]))


@torch.no_grad()
def check_forward(model: TorchCNNBLSTM, timesteps=123):
    model.eval()
    x = torch.randn(2, timesteps, 257, dtype=torch.float32)
    avg, frame = model(x)
    print(f"[Torch] input={tuple(x.shape)} avg={tuple(avg.shape)} frame={tuple(frame.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Convert MOSNet Keras .h5 weights to PyTorch.")
    parser.add_argument(
        "--h5",
        type=str,
        default="pre_trained/cnn_blstm.h5",
        help="Keras .h5 weights path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="pre_trained/cnn_blstm_torch.pt",
        help="Output PyTorch state_dict path",
    )
    args = parser.parse_args()

    model = TorchCNNBLSTM()
    load_keras_h5_weights(model, args.h5)
    check_forward(model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[OK] Saved Torch state_dict to: {out_path}")


if __name__ == "__main__":
    main()
