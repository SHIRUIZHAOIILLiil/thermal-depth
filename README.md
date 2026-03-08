<h1><img src="./img/Iris_logo.png" style="height: 2em; vertical-align: middle;">Iris: Integrating Language into Diffusion-based Monocular Depth Estimation</h1>

<p>
  <a href="https://arxiv.org/abs/2411.16750"><img src="https://img.shields.io/badge/arXiv-2411.16750-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://adonis-galaxy.github.io/Iris-website/"><img src="https://img.shields.io/badge/Project-Website-4a90d9?style=flat&logo=googlechrome&logoColor=white" alt="Project Website"></a>
</p>

<p>
  <a href="https://adonis-galaxy.github.io/homepage/">Ziyao Zeng</a><sup>1*</sup>&ensp;
  <a href="https://jingchengni.com/">Jingcheng Ni</a><sup>2*</sup>&ensp;
  <a href="https://preacherwhite.github.io/">Daniel Wang</a><sup>1</sup>&ensp;
  <a href="https://patrickqrim.github.io/">Patrick Rim</a><sup>1</sup>&ensp;
  <a href="https://fuzzythecat.github.io/">Younjoon Chung</a><sup>1</sup>&ensp;
  <a href="https://vision.cs.yale.edu/members/fred-yang/">Fengyu Yang</a><sup>1</sup>&ensp;
  <a href="https://www.image.cau.ac.kr/people/director">Byung-Woo Hong</a><sup>3</sup>&ensp;
  <a href="https://vision.cs.yale.edu/members/alex-wong/">Alex Wong</a><sup>1</sup>
</p>

<p>
  <sup>1</sup>Yale University &ensp; <sup>2</sup>Brown University &ensp; <sup>3</sup>Korea University
</p>

<p><sup>*</sup>Equal contribution</p>


## Overview

Traditional monocular depth estimation suffers from inherent ambiguity and visual nuisances. We demonstrate that language can enhance monocular depth estimation by providing an additional condition (rather than images alone) aligned with plausible 3D scenes, thereby reducing the solution space for depth estimation. This conditional distribution is learned during the text-to-image pre-training of diffusion models. To generate images under various viewpoints and layouts that precisely reflect textual descriptions, the model implicitly models object sizes, shapes, and scales, their spatial relationships, and the overall scene structure. In this paper, Iris, we investigate the benefits of our strategy to integrate text descriptions into training and inference of diffusion-based depth estimation models. We experiment with three different diffusion-based monocular depth estimators (Marigold, Lotus, and E2E-FT) and their variants. By training on HyperSim and Virtual KITTI, and evaluating on NYUv2, KITTI, ETH3D, ScanNet, and DIODE, we find that our strategy improves the overall monocular depth estimation accuracy, especially in small areas. It also improves the model's depth perception of specific regions described in the text. We find that by providing more details in the text, the depth prediction can be iteratively refined. Simultaneously, we find that language can act as a constraint to accelerate the convergence of both training and the inference diffusion trajectory. Code and generated text data will be released upon acceptance.

---

## Structure

```
Iris-iris/
├── run_train.sh              # Unified training entry script
├── run_eval.sh               # Unified evaluation entry script
├── README.md
│
├── Lotus/                    # Lotus baseline (Discriminative & Generative)
│   ├── train_scripts/
│   │   ├── train_iris_d_depth.sh # Iris for Lotus-D
│   │   └── train_iris_g_depth.sh # Iris for Lotus-G
│   ├── eval_rng.py           
│   ├── eval_scripts/
│   │   ├── eval-depth-d.sh
│   │   └── eval-depth-g.sh
│
├── diffusion-e2e-ft/         # E2E-FT baseline
│   ├── training/
│   │   ├── scripts/
│   │   │   └── train_stable_diffusion_e2e_ft_depth.sh #Iris for E2E-FT
│   ├── experiments/depth/eval_args/stable_diffusion_e2e_ft/
│   │   ├── 0_infer_eval_all.sh
│   │   ├── 11_infer_nyu.sh / 12_eval_nyu.sh
│   │   ├── 21_infer_kitti.sh / 22_eval_kitti.sh
│   │   ├── 31_infer_eth3d.sh / 32_eval_eth3d.sh
│   │   ├── 41_infer_scannet.sh / 42_eval_scannet.sh
│   │   └── 51_infer_diode.sh / 52_eval_diode.sh
│
├── marigold/                 # Marigold baseline           
│   ├── run.sh                # Iris for marigold
│   ├── eval.py / eval.sh     
│   ├── infer.py / run.py     
│
└── InternVL/                 # Text description generation
    └── scripts/
        └── e2e_inference/                
            ├── inference_hypersim.py
        └── lotus_inference/                
            ├── inference_hypersim.py
            ├── inference_vkitti.py
            ├── inference_nyu.py
            ├── inference_kitti.py
            ├── inference_eth3d.py
            ├── inference_scannet.py
            └── inference_diode.py
```

---

## 1. Environment Setup

Each baseline has its own `requirements.txt`. Install for the baseline you want to run:

```bash
# Lotus
cd Lotus
pip install -r requirements.txt

# E2E-FT
cd diffusion-e2e-ft
pip install -r requirements.txt

# Marigold
cd marigold
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

---

## 2. Data Preparation

### 2.1 Training Data

All three baselines are trained on **HyperSim** and **Virtual KITTI**, with different dataset samples according to original [Lotus](https://github.com/EnVision-Research/Lotus), [Diffusion-E2-FT](https://github.com/VisualComputingInstitute/diffusion-e2e-ft), and [Marigold](https://github.com/prs-eth/Marigold), respectively.

### 2.2 Evaluation Data
All three baselines are evaluated on **NYUv2**, **KITTI**, **ETH3D**, **ScanNet**, and**DIODE**.

### 2.3 Text Descriptions

For E2E and Lotus, text descriptions are generated using **InternVL3-8B** (see `InternVL/scripts/e2e_inference` and `InternVL/scripts/lotus_inference`).

Pre-generated JSON files are needed for both training and evaluation:

**Training text:**
- `hypersim_depth_descriptions.json` — Hypersim training images
- `vkitti_depth_descriptions.json` — Virtual KITTI training images

**Evaluation text:**
- `nyu_depth_descriptions.json`
- `kitti_depth_descriptions.json`
- `eth3d_depth_descriptions.json`
- `scannet_depth_descriptions.json`
- `diode_depth_descriptions.json`

For Marigold, text descriptions are stored as `.txt` files under `marigold/Marigold_eval_text/` and `marigold/Marigold_test_text/`.

---

## 3. Training


```bash
# Lotus Discriminative
./run_train.sh lotus d

# Lotus Generative
./run_train.sh lotus g

# E2E-FT
./run_train.sh e2e

# Marigold
./run_train.sh marigold
```

## 4. Evaluation


```bash
# Lotus Discriminative
./run_eval.sh lotus d

# Lotus Generative
./run_eval.sh lotus g

# E2E-FT
./run_eval.sh e2e

# Marigold
./run_eval.sh marigold
```


## Citation

```bibtex
@article{zeng2024iris,
    title={Iris: Integrating Language into Diffusion-based Monocular Depth Estimation},
    author={Zeng, Ziyao and Ni, Jingcheng and Wang, Daniel and Rim, Patrick and Chung, Younjoon and Yang, Fengyu and Hong, Byung-Woo and Wong, Alex},
    journal={arXiv preprint arXiv:2411.16750},
    year={2024}
}
```