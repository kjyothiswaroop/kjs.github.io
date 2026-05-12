---
type: project
title: Deformable Object Manipulation
summary: Deformable BiManipulation — Bimanual Cloth Folding with Imitation Learning.
tags:
  - Isaac Sim
  - VLA
  - Diffusion Policy
  - PyTorch
  - LeRobot
  - SO-101

date: 2026-03-01
weight: 5
share: false
links:
  - name: Code
    url: https://github.com/kjyothiswaroop/lehome-laundrynauts
    icon: github
    icon_pack: fab
author: Jyothi Swaroop
video: https://www.youtube.com/embed/m24g43MBf3A
project_title: 'LeHome Challenge (ICRA 2026): Bimanual Cloth Folding via Imitation Learning'
---

-----------------------------------------------------------------
## Overview

The **leHome Challenge** (ICRA 2026) is a **Deformable BiManipulation Challenge** focused on folding laundry using a bimanual **SO-101** robot. Teams must train policies capable of manipulating deformable objects — one of the hardest open problems in robot learning due to the near-infinite configuration space of cloth.

We finished **54th out of 230+ teams**.

-----------------------------------------------------------------
## Data Collection

Automatic data collection was attempted by extracting **privileged information** about the cloth mesh directly from the **Isaac Sim** scene and passing goal poses to **cuRobo** for bimanual IK solving and motion planning. In practice this pipeline was too brittle — cuRobo IK failures and sim–real cloth geometry mismatches meant trajectories rarely transferred to usable demonstrations.

We fell back to **manual teleoperation** on the physical SO-101 arms, then significantly expanded the dataset through **data augmentation**: applying colour-jitter, brightness, contrast, and blur filters to camera observations to improve policy generalisation across lighting conditions.

-----------------------------------------------------------------
## Policy

Five policy architectures were trained and evaluated:

| Policy | Notes |
|--------|-------|
| **ACT** | Action Chunking with Transformers |
| **Diffusion Policy** | Denoising diffusion over action sequences |
| **SmolVLA** | Vision-Language-Action model — **best performance** |
| **LingBotVLA** | Language-conditioned VLA |

**SmolVLA** achieved the highest task success rate across our evaluations. All policies were trained using the **LeRobot** framework with **PyTorch**.

-----------------------------------------------------------------
## Results

Ranked **54th / 230+ teams** at the leHome Challenge (ICRA 2026).
