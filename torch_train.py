#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import utils
from torch_model import build_model

random.seed(1984)
np.random.seed(1984)
torch.manual_seed(1984)

DATA_DIR = './data'
BIN_DIR = os.path.join(DATA_DIR, 'bin')
OUTPUT_DIR = './output'
NUM_TRAIN = 13580
NUM_TEST = 4000
NUM_VALID = 3000


class MosBinDataset(Dataset):
    def __init__(self, rows, bin_root):
        self.rows = rows
        self.bin_root = bin_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        audio, mos = self.rows[idx].split(',')
        filename = audio.split('.')[0]
        feat = utils.read(os.path.join(self.bin_root, filename + '.h5'))['mag_sgram'][0]
        return torch.from_numpy(feat).float(), torch.tensor(float(mos), dtype=torch.float32), audio


def collate_batch(batch):
    feats, mos, names = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in feats], dtype=torch.long)
    max_t = int(lengths.max().item())
    padded = torch.zeros(len(feats), max_t, feats[0].shape[1], dtype=torch.float32)
    mask = torch.zeros(len(feats), max_t, 1, dtype=torch.float32)
    for i, feat in enumerate(feats):
        t = feat.shape[0]
        padded[i, :t] = feat
        mask[i, :t, 0] = 1.0
    mos = torch.stack(mos)
    return padded, mos, mask, list(names)


def frame_loss(frame_pred, mos, mask):
    target = mos[:, None, None].expand_as(frame_pred)
    se = (frame_pred - target) ** 2
    se = se * mask
    denom = mask.sum().clamp_min(1.0)
    return se.sum() / denom


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for feats, mos, mask, _ in loader:
        feats = feats.to(device)
        mos = mos.to(device)
        mask = mask.to(device)

        if training:
            optimizer.zero_grad()

        avg_pred, frame_pred = model(feats)
        loss_avg = ((avg_pred.squeeze(-1) - mos) ** 2).mean()
        loss_frame = frame_loss(frame_pred, mos, mask)
        loss = loss_avg + loss_frame

        if training:
            loss.backward()
            optimizer.step()

        total += loss.item() * feats.size(0)
        count += feats.size(0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate_and_plot(model, test_rows, device):
    model.eval()
    mos_predict = np.zeros([len(test_rows), ])
    mos_true = np.zeros([len(test_rows), ])
    records = []

    for i in tqdm(range(len(test_rows))):
        audio, mos = test_rows[i].split(',')
        filename = audio.split('.')[0]
        mag = utils.read(os.path.join(BIN_DIR, filename + '.h5'))['mag_sgram']
        x = torch.from_numpy(mag).float().to(device)
        avg, _ = model(x)
        mos_predict[i] = float(avg.squeeze().cpu().item())
        mos_true[i] = float(mos)
        records.append({'audio': audio, 'true_mos': mos_true[i], 'predict_mos': mos_predict[i]})

    df = pd.DataFrame(records)

    plt.style.use('seaborn-v0_8-deep')
    x = df['true_mos']
    y = df['predict_mos']
    bins = np.linspace(1, 5, 40)
    plt.figure(2)
    plt.hist([x, y], bins, label=['true_mos', 'predict_mos'])
    plt.legend(loc='upper right')
    plt.xlabel('MOS')
    plt.ylabel('number')
    plt.savefig('./output/MOSNet_distribution.png', dpi=150)

    mse = np.mean((mos_true - mos_predict) ** 2)
    lcc = np.corrcoef(mos_true, mos_predict)
    srcc = scipy.stats.spearmanr(mos_true.T, mos_predict.T)
    print('[UTTERANCE] Test error= %f' % mse)
    print('[UTTERANCE] Linear correlation coefficient= %f' % lcc[0][1])
    print('[UTTERANCE] Spearman rank correlation coefficient= %f' % srcc[0])

    m = np.max([np.max(mos_predict), 5])
    plt.figure(3)
    plt.scatter(mos_true, mos_predict, s=15, color='b', marker='o', edgecolors='b', alpha=.20)
    plt.xlim([0.5, m])
    plt.ylim([0.5, m])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}'.format(lcc[0][1], srcc[0], mse))
    plt.savefig('./output/MOSNet_scatter_plot.png', dpi=150)

    sys_df = pd.read_csv(os.path.join(DATA_DIR, 'vcc2018_system.csv'))
    df['system_ID'] = df['audio'].str.split('_').str[-1].str.split('.').str[0] + '_' + df['audio'].str.split('_').str[0]
    result_mean = df[['system_ID', 'predict_mos']].groupby(['system_ID']).mean()
    mer_df = pd.merge(result_mean, sys_df, on='system_ID')

    sys_true = mer_df['mean']
    sys_predicted = mer_df['predict_mos']

    mse = np.mean((sys_true - sys_predicted) ** 2)
    lcc = np.corrcoef(sys_true, sys_predicted)
    srcc = scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
    print('[SYSTEM] Test error= %f' % mse)
    print('[SYSTEM] Linear correlation coefficient= %f' % lcc[0][1])
    print('[SYSTEM] Spearman rank correlation coefficient= %f' % srcc[0])

    m = np.max([np.max(sys_predicted), 5])
    plt.figure(4)
    plt.scatter(sys_true, sys_predicted, s=25, color='b', marker='o', edgecolors='b')
    plt.xlim([1, m])
    plt.ylim([1, m])
    plt.xlabel('True MOS')
    plt.ylabel('Predicted MOS')
    plt.title('LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}'.format(lcc[0][1], srcc[0], mse))
    plt.savefig('./output/MOSNet_system_scatter_plot.png', dpi=150)


def load_checkpoint_weights(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model to train with, CNN, BLSTM or CNN-BLSTM')
    parser.add_argument('--epoch', type=int, default=100, help='number epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='number batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if not args.model:
        raise ValueError('please specify model to train with, CNN, BLSTM or CNN-BLSTM')

    print('training with model architecture: {}'.format(args.model))
    print('epochs: {}\\nbatch_size: {}'.format(args.epoch, args.batch_size))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mos_list = utils.read_list(os.path.join(DATA_DIR, 'mos_list.txt'))
    random.shuffle(mos_list)

    train_list = mos_list[0:-(NUM_TEST + NUM_VALID)]
    random.shuffle(train_list)
    valid_list = mos_list[-(NUM_TEST + NUM_VALID):-NUM_TEST]
    test_list = mos_list[-NUM_TEST:]

    print('{} for training; {} for valid; {} for testing'.format(NUM_TRAIN, NUM_TEST, NUM_VALID))

    train_loader = DataLoader(MosBinDataset(train_list, BIN_DIR), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(MosBinDataset(valid_list, BIN_DIR), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device(args.device)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    best_path = os.path.join(OUTPUT_DIR, 'mosnet_torch.pt')

    for epoch in range(1, args.epoch + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        valid_loss = run_epoch(model, valid_loader, None, device)
        print(f'[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={valid_loss:.6f}')

        if valid_loss < best_val:
            best_val = valid_loss
            torch.save({
                'epoch': epoch,
                'model_name': args.model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': valid_loss,
            }, best_path)
            print(f'[Checkpoint] saved: {best_path}')

    load_checkpoint_weights(model, best_path, device)
    print('testing...')
    evaluate_and_plot(model, test_list, device)


if __name__ == '__main__':
    main()
