---
type: project
title: Grayscale Image Colourization
summary: Grayscale image colorization via a diffusion model built from scratch.
tags:
  - PyTorch
  - Diffusion Models
  - Computer Vision
  - Deep Learning
  - HuggingFace
  - Generative AI

date: 2025-06-01
weight: 5
share: false
links:
  - name: Code
    url: https://github.com/kjyothiswaroop/Conditional-Diffusion-for-Grayscale-Image-Colorization
    icon: github
    icon_pack: fab
author: Jyothi Swaroop
project_title: 'Conditional Diffusion for Grayscale Image Colorization'
---

-----------------------------------------------------------------
## Overview

A **conditional diffusion model** for grayscale image colorization, built entirely from scratch in PyTorch — forward noising process, noise schedule, U-Net, EMA, and reverse diffusion loop. The grayscale image acts as the conditioning signal, concatenated as an additional channel to the noisy RGB image at each denoising step.

-----------------------------------------------------------------
## Dataset

Source images come from the **CelebA-HQ** dataset (`korexyz/celeba-hq-256x256`). Each image is resized to **128×128** and paired with its grayscale version. The resulting dataset is pushed to HuggingFace (`kjswaroopNU/celebahq-128-gray`) and loaded directly during training.

| Split | Samples |
|-------|---------|
| Train | 28,000 |
| Validation | 1,000 |
| Test | 1,000 |

-----------------------------------------------------------------
## Forward Process

The forward noising process uses an **offset cosine schedule** over T = 1000 timesteps. Signal and noise rates satisfy the identity signal² + noise² = 1:

$$x_t = \text{signal\_rate}(t) \cdot x_0 + \text{noise\_rate}(t) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

The schedule interpolates angles between a `max_signal_rate` of 0.95 and `min_signal_rate` of 0.02, keeping signal-to-noise well-behaved at both ends.

-----------------------------------------------------------------
## U-Net

The U-Net is written from scratch with the following structure:

| Block | Details |
|-------|---------|
| **DownBlock** | 2× ResidualBlock + AvgPool2d |
| **Bottleneck** | 2× ResidualBlock |
| **UpBlock** | Bilinear upsample + 2× ResidualBlock with skip connections |
| **Activation** | SiLU throughout |

The noise variance is injected via a **sinusoidal embedding** (log-spaced frequencies, sin + cos concatenated), upsampled to 128×128 and concatenated after the first convolution. The grayscale conditioning is concatenated to the noisy RGB input, giving the network 4 input channels.

-----------------------------------------------------------------
## Training

The network is trained to predict the noise $\epsilon$ added at each timestep (MSE loss). An **EMA copy** of the U-Net (decay = 0.999) is maintained throughout and used exclusively at inference time for smoother outputs.

A subtle bug was caught during development: BatchNorm **buffers** (running mean/variance) are not touched by `model.parameters()`, so the EMA network was computing fresh batch statistics at inference instead of using the accumulated training stats. The fix copies buffers explicitly alongside weight averaging each step.

![](result.png)

-----------------------------------------------------------------
## Gradio UI

Running `python src/eval.py` launches a **Gradio web app** for interactive inference. Upload any grayscale image, set the number of diffusion steps (10–100), and the number of colorized samples to generate (1–8). The EMA network runs the full reverse diffusion loop and returns the colorized outputs in a gallery — no code required.

-----------------------------------------------------------------
## MLOps

- **Hydra** — all hyperparameters live in `configs/config.yaml` and are overridable from the CLI (`python src/train.py lr=0.0001 batch_size=64`), making every run reproducible without touching source code.
- **Weights & Biases** — per-step loss, per-epoch loss, and sample grids (grayscale / generated / ground truth) logged every 10 epochs.
- Checkpoints saved every 50 epochs; training resumes from any checkpoint via `resume=true`.
