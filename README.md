# MOSNet
Implementation of  "MOSNet: Deep Learning based Objective Assessment for Voice Conversion"
https://arxiv.org/abs/1904.08352

## Dependency
Linux Ubuntu 16.04
- GPU: GeForce RTX 2080 Ti
- Driver version: 418.67
- CUDA version: 10.1

Python 3.12
- torch
- scipy
- pandas
- matplotlib
- librosa
- h5py

### Environment set-up
For example,
```
conda create -n mosnet python=3.5
conda activate mosnet
pip install -r requirements.txt
conda install cudnn=7.6.0
```

## Usage

### Reproducing results in the paper (PyTorch default)

1. `cd ./data` and run `bash download.sh` to download the VCC2018 evaluation results and submitted speech. (downsample the submitted speech might take some times)
2. Run `python mos_results_preprocess.py` to prepare the evaluation results. (Run `python bootsrap_estimation.py` to do the bootstrap experiment for intrinsic MOS calculation)
3. Run `python utils.py` to extract .wav to .h5
4. Run `python train.py --model CNN-BLSTM` to train a CNN-BLSTM version of MOSNet with PyTorch. (`CNN`, `BLSTM`, `CNN-BLSTM` are supported in `torch_model.py`)
5. Run `python test.py` to test on the pre-trained PyTorch weights (`pre_trained/cnn_blstm_torch.pt`).


#### Note
The default implementation in this repository is now PyTorch-based. Legacy TensorFlow/Keras files are moved under `legacy_tf/` for reference only.

### Evaluating your custom waveform samples

1. Put the waveforms you wish to evaluate in a folder. For example, `<path>/<to>/<samples>`
2. Run `python ./custom_test.py --rootdir <path>/<to>/<samples>`

This script will evaluate all the `.wav` files in `<path>/<to>/<samples>`, and write the results to `<path>/<to>/<samples>/MOSnet_result_raw.txt`. By default, the `pre_trained/cnn_blstm_torch.pt` pretrained model is used. If you wish to use another checkpoint, specify `--pretrained_model`.

### Legacy TensorFlow

Original TensorFlow/Keras implementation files are kept under `legacy_tf/`.

## Citation

If you find this work useful in your research, please consider citing:
```
@inproceedings{mosnet,
  author={Lo, Chen-Chou and Fu, Szu-Wei and Huang, Wen-Chin and Wang, Xin and Yamagishi, Junichi and Tsao, Yu and Wang, Hsin-Min},
  title={MOSNet: Deep Learning based Objective Assessment for Voice Conversion},
  year=2019,
  booktitle={Proc. Interspeech 2019},
}
```
 
 
## License

This work is released under MIT License (see LICENSE file for details).


## VCC2018 Database & Results

The model is trained on the large listening evaluation results released by the Voice Conversion Challenge 2018.<br>
The listening test results can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3257)<br>
The databases and results (submitted speech) can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3061)<br>
