#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import torch
from tqdm import tqdm

import utils
from torch_model import TorchCNNBLSTM

random.seed(1984)

DATA_DIR = './data'
BIN_DIR = os.path.join(DATA_DIR, 'bin')
PRE_TRAINED_DIR = './pre_trained'
OUTPUT_DIR = './output'
NUM_TEST = 4000
NUM_VALID = 3000


def load_weights(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)


@torch.no_grad()
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mos_list = utils.read_list(os.path.join(DATA_DIR, 'mos_list.txt'))
    random.shuffle(mos_list)
    test_list = mos_list[-NUM_TEST:]

    print('{} for training; {} for valid; {} for testing'.format(len(mos_list) - NUM_TEST - NUM_VALID, NUM_VALID, NUM_TEST))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TorchCNNBLSTM().to(device)
    ckpt = os.path.join(PRE_TRAINED_DIR, 'cnn_blstm_torch.pt')
    load_weights(model, ckpt, device)
    model.eval()

    print('testing...')
    mos_predict = np.zeros([len(test_list), ])
    mos_true = np.zeros([len(test_list), ])
    records = []

    for i in tqdm(range(len(test_list))):
        filepath = test_list[i].split(',')
        filename = filepath[0].split('.')[0]
        mag = utils.read(os.path.join(BIN_DIR, filename + '.h5'))['mag_sgram']

        avg, _ = model(torch.from_numpy(mag).float().to(device))
        mos_predict[i] = float(avg.squeeze().cpu().item())
        mos_true[i] = float(filepath[1])
        records.append({'audio': filepath[0], 'true_mos': mos_true[i], 'predict_mos': mos_predict[i]})

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


if __name__ == '__main__':
    main()
