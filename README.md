# Multi-Scale Hierarchical Adaptive No-Reference Image Quality Assessment via Diffusion Prior

![Framework](/figures/framework.jpg)
 
Abstract: *Objective Real-world image distortions exhibit multi-scale and spatially non-uniform characteristics, which pose significant challenges for no-reference image quality assessment (NR-IQA). Existing NR-IQA methods struggle to simultaneously capture global semantics and local textures due to inherent architectural limitations. Convolutional neural network (CNN) based methods suffer from restricted receptive fields, making them inadequate for modeling long-range dependencies and handling non-uniform distortions across scales. Transformer-based approaches, while capable of global semantic integration, lack sufficient sensitivity to fine-grained local distortions and incur high computational costs. Moreover, fixed multi-stage feature fusion strategies commonly adopted in existing methods cannot adapt to diverse image contents, leading to limited prediction accuracy, poor subjective consistency, and restricted cross-dataset generalization. To overcome these limitations, this study develops a novel NR-IQA method that leverages powerful visual priors of pre-trained diffusion models while establishing adaptive feature encoding and fusion mechanisms tailored to content characteristics. Method A multi-scale hierarchical adaptive NR-IQA method based on diffusion priors is proposed. The method employs a pre-trained text-to-image diffusion model (Stable Diffusion v1.5) as the backbone. The framework consists of three core modules: multi-scale feature extraction, quality conditional embedding, and hierarchical adaptive quality decoding. In the multi-scale feature extraction module, the input image is first encoded into latent space representations through a variational autoencoder (VAE) to obtain compact global semantic features. However, VAE encoding inevitably loses high-frequency textures, edge details, and local distortion information during downsampling. To compensate for this loss, a multi-level feature extraction branch (MFEB) is constructed in the pixel space, extracting multi-scale residual features aligned with the downsampling stages of the denoising U-Net. The MFEB comprises four cascaded adapter blocks with channel attention and spatial gating mechanisms. A frequency-aware enhancement module (FEM) is embedded after each adapter block, employing depthwise separable convolution with center-initialized kernels to isolate high-frequency residual components, effectively enhancing perception of blur and compression artifacts. Cross-scale attention fusion modules (CAFM) are introduced between adjacent adapter blocks to promote multi-scale information interaction through spatial and channel attention mechanisms. The extracted residual features are injected into the downsampling path of the U-Net as additive residuals. In the quality conditional embedding module, predefined quality description texts covering continuous quality levels are encoded by a frozen CLIP text encoder to obtain initial embeddings. A lightweight text adapter with residual connections is applied for adaptation. The adapted embeddings are replicated along the batch dimension and fed into cross-attention layers of the U-Net to guide conditional denoising in quality-aware feature space. In the hierarchical adaptive quality decoding module, feature maps from four stages of the U-Net upsampling path are unified to the same spatial resolution through bicubic interpolation and projected to 512-dimensional channel space. Squeeze-and-Excitation (SE) modules refine the channels. A layer adaptive attention fusion (LAAF) mechanism dynamically allocates aggregation weights for the four stages based on global content descriptors. Global average pooling extracts descriptors from each stage, which are concatenated and processed by a lightweight multilayer perceptron to learn optimal hierarchical combinations. The generated weights are normalized by Softmax and applied for element-wise weighted fusion of refined stage features. The fused features undergo further convolution and channel refinement before quality score regression through a three-layer multilayer perceptron. Result Extensive experiments are conducted on four widely recognized real-world distortion datasets: CLIVE, KonIQ-10k, LIVEFB, and SPAQ The proposed method is compared against nine state-of-the-art approaches: HyperIQA, CLIP-IQA, Q-Align, SaTQA, ReIQA, LoDa, DP-IQA, and MDM-GFIQA. Evaluation metrics include Pearson linear correlation coefficient (PLCC) and Spearman rank-order correlation coefficient (SRCC), where higher absolute values indicate stronger correlation. Experimental results demonstrate optimal or suboptimal PLCC and SRCC across all datasets. On KonIQ-10k, PLCC reaches 0.955 and SRCC reaches 0.946, surpassing the second-best method (DP-IQA: PLCC 0.949, SRCC 0.941) by 0.63% and 0.53%. On CLIVE, PLCC reaches 0.924 and SRCC reaches 0.907, outperforming the second-best (DP-IQA: PLCC 0.911, SRCC 0.891) by 1.43% and 1.80%. On LIVEFB, PLCC reaches 0.706 and SRCC reaches 0.589, exceeding the second-best (DP-IQA: PLCC 0.681, SRCC 0.578) by 3.67% and 1.90%. On SPAQ, PLCC reaches 0.927 and SRCC reaches 0.924, slightly below Q-Align (PLCC 0.933, SRCC 0.930) but comparable to LoDa and superior to other compared methods. Ablation experiments on KonIQ-10k and CLIVE verify synergistic effects of proposed modules. The baseline with frozen U-Net encoder achieves PLCC/SRCC of 0.949/0.941 on KonIQ-10k and 0.911/0.891 on CLIVE. Adding the frequency-aware enhancement (FA) module improves PLCC to 0.951 and 0.914 respectively. Adding FA with cross-scale attention (CSA) achieves 0.953/0.945 and 0.921/0.902. The complete model with layer adaptive attention fusion (LAAF) achieves best performance of 0.955/0.946 and 0.924/0.907. Cross-dataset generalization experiments among KonIQ-10k, CLIVE, and LIVEFB demonstrate favorable average cross-domain performance, with SRCC of 0.781 for LIVEFB-to-KonIQ transfer and 0.856 for KonIQ-to-CLIVE transfer. Conclusion This study proposes a multi-scale hierarchical adaptive NR-IQA method based on diffusion priors. By synergistically extracting global semantics through VAE latent encoding and compensating local high-frequency details through the multi-level feature extraction branch, and dynamically aggregating multi-stage features through the hierarchical adaptive attention fusion mechanism, the proposed method effectively improves prediction accuracy, subjective consistency, and cross-dataset generalization for real-world image quality assessment. Experimental results on multiple public datasets validate superiority over state-of-the-art approaches, and ablation studies confirm contributions of each core module. This work provides an effective solution for no-reference image quality assessment in real-world scenarios.*

# Preparation

**Environments**

We recommend installing 64-bit Python 3.11 and PyTorch 2.6.0. On a CUDA GPU machine, the following will do the trick:

```
pip install -r requirements.txt
```

We have done all testing and development using an A100 GPU.

**Download required files**

[CLIP](https://github.com/openai/CLIP). Place the **"clip"** folder in this project.

**Download datasets**

**-KonIQ-10K.** Download the [KonIQ-10k](https://osf.io/hcsdy/) dataset (OSF Storage -> database -> 1024x768). Make sure the path of its .csv file is 'data/koniq/koniq10k_distributions_sets.csv', and the root path of images is 'data/koniq/1024x768' in your project.

**-CLIVE.** Download the [CLIVE](https://live.ece.utexas.edu/research/ChallengeDB/index.html) dataset. Make sure the root path of its .mat files is 'data/ChallengeDB_release/Data', and the root path of images is 'data/ChallengeDB_release/Images' in your project.

**-LIVEFB.** Please refer to [FLIVE-dataset](https://github.com/niu-haoran/FLIVE_Database/tree/master). Make sure the root path of its .csv file is 'data/livefb_database/labels_image.csv', and the root path of images is 'data/livefb_database' in your project.

**-SPAQ.** Please refer to [Perceptual Quality Assessment of Smartphone Photography](https://github.com/h4nwei/SPAQ). Make sure the root path of its .xlsx file is 'data/spaq/MOS and Image attribute scores.xlsx', and the root path of images is 'data/spaq/SPAQ/TestImage' in your project.

# Train

1. Generate conditional text embeddings

```
python gene_text_embedding.py
```
This produces 'quality_embeddings.pth' in the project root.

2. Train SDA-IQA models. Use '--dataset' to switch between datasets (koniq, clive, livefb, spaq)

```
train_sda_iqa.py --dataset clive --adapter_variant all --epochs 12
```
Optional arguments:
--adapter_variant: ablation variant corresponding to the paper ('baseline', 'fa', 'fa+csa', 'laaf', 'all'), where FA is the frequency-aware enhancement module, CSA is the cross-scale attention fusion module, and LAAF is the layer adaptive attention fusion in the decoder. 'all' enables FA + CSA + LAAF.
--accumulation_steps: gradient accumulation steps. Set >1 (e.g. 2) when GPU memory is limited.
--sd_path, --class_embedding_path: custom paths of the Stable Diffusion weights and the text embeddings.

Please note that, to reduce the overall runtime of the training script, validation is set by default to occur after each training epoch. However, due to the small batch size, the model undergoes frequent updates within each epoch, often reaching its optimal performance midway through a certain epoch, followed by a slight decline in performance. To ensure the best model is captured, it is recommended to perform validation every 250 training steps or fewer.

3. Training randomly splits each dataset into 80% training and 20% testing, and saves the split indices as '{dataset}_train_indices.pth' and '{dataset}_test_indices.pth'. To reproduce the exact results reported in the paper, please use the provided split files instead of re-splitting. The split files for all four datasets are only a few kilobytes each and are provided together with the checkpoint (https://drive.google.com/uc?export=download&id=1mla5G73y8CPawUQNv9Pv0i-z7x4nEz2P and https://drive.google.com/uc?export=download&id=1jivVUOOJoQ3R-xX3oynGWWtjeeKdWhPX). Set load_indices=True in the training script to reuse them (test_sda_iqa.py loads them by default).


# Checkpoints

Due to cloud storage limitations, we currently release the checkpoint trained on CLIVE, together with the train/test split files for all four datasets:
CLIVE. [https://drive.google.com/uc?export=download&id=1w57BKtkwmpaOdosgPro-HNEziY3pCljq]
With this checkpoint and the released test_sda_iqa.py script, the complete evaluation pipeline can be verified end-to-end. Checkpoints for the other three datasets (KonIQ-10K, LIVEFB, SPAQ) can be readily reproduced by running the training script with the provided split files and configurations, and are also available from the authors upon request.

Acknowledgement
This repository is built upon DP-IQA. We thank the authors for open-sourcing their code.

License
This project is licensed under CC BY-NC-SA 4.0 and is released for academic research use only. See LICENSE for details.