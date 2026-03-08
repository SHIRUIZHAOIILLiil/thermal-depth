# Iris: Integrating Language into Diffusion-based Monocular Depth Estimation #

Official implementation of the paper "Iris: Integrating Language into Diffusion-based Monocular Depth Estimation"


[Paper Link](https://arxiv.org/abs/2411.16750)


Authors: Ziyao Zeng*, Jingcheng Ni*, Daniel Wang, Patrick Rim, Younjoon Chung, Fengyu Yang, Byung-Woo Hong, Alex Wong

## Overview ##
Traditional monocular depth estimation suffers from inherent ambiguity and visual nuisances. We demonstrate that language can enhance monocular depth estimation by providing an additional condition (rather than images alone) aligned with plausible 3D scenes, thereby reducing the solution space for depth estimation. This conditional distribution is learned during the text-to-image pre-training of diffusion models. To generate images under various viewpoints and layouts that precisely reflect textual descriptions, the model implicitly models object sizes, shapes, and scales, their spatial relationships, and the overall scene structure. In this paper, Iris, we investigate the benefits of our strategy to integrate text descriptions into training and inference of diffusion-based depth estimation models. We experiment with three different diffusion-based monocular depth estimators (Marigold, Lotus, and E2E-FT) and their variants. By training on HyperSim and Virtual KITTI, and evaluating on NYUv2, KITTI, ETH3D, ScanNet, and DIODE, we find that our strategy improves the overall monocular depth estimation accuracy, especially in small areas. It also improves the model's depth perception of specific regions described in the text. We find that by providing more details in the text, the depth prediction can be iteratively refined. Simultaneously, we find that language can act as a constraint to accelerate the convergence of both training and the inference diffusion trajectory. Code and generated text data will be released upon acceptance.

## Setup Environment ##
Create Virtual Environment:
```
virtualenv -p /usr/bin/python3.8 ~/venvs/iris

vim  ~/.bash_profile
```
Insert the following line to vim:
```
alias priordiffusion="export CUDA_HOME=/usr/local/cuda-11.1 && source ~/venvs/iris/bin/activate"
```
Then activate it, install all packages:
```
source ~/.bash_profile

iris

pip install -r requirements.txt
```

Please follow [Marigold](https://github.com/prs-eth/Marigold) to set up training data. For text generation, we use [LLaVA](https://github.com/haotian-liu/LLaVA). You can also use [InternVL3.5](https://internvl.github.io/blog/2025-08-26-InternVL-3.5/) for better text quality.


### Training and Evaluation ###
After organizing the training images and text, to train the model, run:
```
sh run.sh
```
After finishing training, to evaluate, run:
```
sh eval.sh
```

## Acknowledgements ##
We would like to acknowledge the use of code snippets from various open-source libraries and contributions from the online coding community, which have been invaluable in the development of this project. Specifically, we would like to thank the authors and maintainers of the following resources:

[CLIP](https://github.com/openai/CLIP)

[Marigold](https://github.com/prs-eth/Marigold)

[LLaVA](https://github.com/haotian-liu/LLaVA)
