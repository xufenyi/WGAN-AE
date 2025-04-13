# WGAN-AE

This repository is part of my undergraduate thesis project, where I implemented a [Wasserstein Generative Adversarial Network (WGAN)](https://arxiv.org/abs/1701.07875) for analysing the acoustic emission (AE) data. The goal of the WGAN is to generate signals that resemble the real AE data. The generated signals can be further used for various purposes such as data augmentation.

The WGAN is implemented using [TensorFlow](https://tensorflow.org). So far, the code has been tested on Windows.

## Operations

1. Ensure the required packages are installed:
```bash
pip install -r requirements.txt
```

2. Train the WGAN:
```bash
python train.py
```

3. (Optional) Visualize the training process with TensorBoard:
```bash
tensorboard --logdir=logs
```
