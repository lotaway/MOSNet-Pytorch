#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import fnmatch
import os

import numpy as np
import torch
from tqdm import tqdm

import utils
from torch_model import TorchCNNBLSTM


def find_files(root_dir, query='*.wav', include_root_dir=True):
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + '/', '') for file_ in files]
    return files


def load_weights(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Evaluate custom waveform files using pretrained MOSNet (PyTorch).')
    parser.add_argument('--rootdir', default=None, type=str, help='rootdir of the waveforms to be evaluated')
    parser.add_argument('--pretrained_model', default='./pre_trained/cnn_blstm_torch.pt', type=str, help='pretrained model file (.pt)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='inference device')
    args = parser.parse_args()

    wavfiles = sorted(find_files(args.rootdir, '*.wav'))
    device = torch.device(args.device)

    print('Loading model weights')
    model = TorchCNNBLSTM().to(device)
    load_weights(model, args.pretrained_model, device)
    model.eval()

    print('Start evaluating {} waveforms...'.format(len(wavfiles)))
    results = []

    for wavfile in tqdm(wavfiles):
        mag_sgram = utils.get_spectrograms(wavfile)
        timestep = mag_sgram.shape[0]
        mag_sgram = np.reshape(mag_sgram, (1, timestep, utils.SGRAM_DIM)).astype(np.float32)

        avg, _ = model(torch.from_numpy(mag_sgram).to(device))
        score = float(avg.squeeze().cpu().item())
        results.append(wavfile + ' {:.3f}'.format(score))

    average = np.mean(np.array([float(line.split(' ')[-1]) for line in results]))
    print('Average: {}'.format(average))

    resultrawpath = os.path.join(args.rootdir, 'MOSnet_result_raw.txt')
    with open(resultrawpath, 'w') as outfile:
        outfile.write('\n'.join(sorted(results)))
        outfile.write('\nAverage: {}\n'.format(average))


if __name__ == '__main__':
    main()
