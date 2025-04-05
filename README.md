# AdsMT: Multimodal Transformer for Predicting Global Minimum Adsorption Energy

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2312.13136-b31b1b.svg)](https://arxiv.org/abs/2312.13136) -->
[![DOI](https://img.shields.io/badge/DOI-10.1038/s41467--025--58499--7-blue.svg)](https://www.nature.com/articles/s41467-025-58499-7)

AdsMT is a novel multi-modal transformer to rapidly predict the global minimum adsorption energy (GMAE) of diverse catalyst/adsorbate combinations based on surface graphs and adsorbate feature vectors without any binding information.

<!-- The fast assessment of the global minimum adsorption energy (GMAE) between catalyst surfaces and adsorbates is crucial for large-scale catalyst screening. However, multiple adsorption sites and numerous possible adsorption configurations for each surface/adsorbate combination make it prohibitively expensive to calculate the GMAE through density functional theory (DFT). Thus, we designed a novel multi-modal transformer called AdsMT to rapidly predict the GMAE based on surface graphs and adsorbate feature vectors without any binding information. -->
<!-- Three diverse benchmark datasets were constructed for this challenging GMAE prediction task. Our AdsMT framework demonstrates excellent performance by adopting the tailored graph encoder and transfer learning, achieving mean absolute errors of 0.09, 0.14, and 0.39 eV, respectively. Beyond GMAE prediction, AdsMT's cross-attention scores showcase the interpretable potential to identify the most energetically favorable adsorption sites. Additionally, uncertainty quantification was integrated into AdsMT to further enhance its trustworthiness in experimental catalyst discovery. -->


## 🚀 Environment Setup
- System requirements: This package requires a standard Linux computer with GPU (supports CUDA >= 11) and enough RAM (> 2 GB). The codes have been tested on NVIDIA RTX 3090, A6000 and A100 GPUs. If you want to run the code on a GPU that does not support CUDA>=11, you need to modify the versions of PyTorch and CUDA in the [env.yml](env.yml) file.
- We'll use `conda` to install dependencies and set up the environment for a Nvidia GPU machine.
We recommend using the [Miniconda installer](https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html).
- After installing `conda`, install [`mamba`](https://mamba.readthedocs.io/en/latest/) to the base environment. `mamba` is a faster, drop-in replacement for `conda`:
    ```bash
    conda install mamba -n base -c conda-forge
    ```
- Then create a conda environment and install the dependencies:
    ```bash
    mamba env create -f env.yml
    ```
    Activate the conda environment with `conda activate adsmt`. It will take about 10 minutes to configure the environment for running code.

## 📌 Datasets
Dataset links: [Zenodo](https://doi.org/10.5281/zenodo.12104162) and [Figshare](https://doi.org/10.6084/m9.figshare.25966573)

We built three GMAE benchmark datasets named `OCD-GMAE`, `Alloy-GMAE` and `FG-GMAE` from [OC20-Dense](https://doi.org/10.1038/s41524-023-01121-5), [Catalysis Hub](https://doi.org/10.1038/s41597-019-0080-z), and [FG-dataset](https://doi.org/10.1038/s43588-023-00437-y) datasets through strict data cleaning, and each data point represents a unique combination of catalyst surface and adsorbate.

| Dataset | Combination Num. | Surface Num. | Adsorbate Num. | Range of GMAE (eV) |
|:--------:|:---------:|:----------:|:-----------:|:------:|
| Alloy-GMAE | 11,260 | 1,916 (37) | 12 (5) | -4.3 $\sim$ 9.1  |
| FG-GMAE | 3,308 | 14 (14)| 202 (5) | -4.0 $\sim$ 0.8  |
| OCD-GMAE | 973 | 967 (54) | 74 (4) | -8.0 $\sim$ 6.4  |

Note: The values in brackets represent the numbers of element types.


We can run [`scripts/download_datasets.sh`](scripts/download_datasets.sh) to download all datasets:
```bash
bash scripts/download_datasets.sh
```

## 🔥 Model Training

### 1. Training from scratch
To train a AdsMT model with different graph encoder on a dataset by [`scripts/train.sh`](scripts/train.sh) and the following command:
```bash
bash scripts/train.sh [DATASET] [GRAPH_ENCODER]
```
This code repo includes 7 different graph encoders:
[SchNet](https://arxiv.org/abs/1706.08566) (schnet), 
[CGCNN](https://doi.org/10.1103/PhysRevLett.120.145301) (cgcnn), 
[DimeNet++](https://arxiv.org/abs/2011.14115) (dpp),
[GemNet-OC](https://arxiv.org/abs/2204.02782) (gemnet-oc), 
[TorchMD-NET](https://arxiv.org/abs/2202.02541) (et), 
[eSCN](https://arxiv.org/abs/2302.03655) (escn), 
AdsGT (adsgt, this work). The log file including experiment results will be found in `exp_results/[DATASET]/[GRAPH_ENCODER].log`. It will take 3-24 hours for one task, depending on the dataset and graph encoder.

### 2. Pretraining on the OC20-LMAE dataset
We provide scripts for model pretraining on the OC20-LMAE dataset. For example, a AdsMT model with different graph encoders will be pretrained by running:
```bash
bash scripts/pretrain_base.sh [GRAPH_ENCODER]
```
The checkpoint file of pretrained model can be found at `checkpoint_dir` in the log file.

### 3. Finetuning on the GMAE datasets
To finetune a AdsMT model on a GMAE dataset, you need to change the `ckpt_path` parameter in the model's configuration file (`configs/[DATASET]/finetune/[GRAPH_ENCODER].yml`) to the checkpoint path of your pre-trained model, then run the following command:
```bash
bash scripts/finetune.sh [DATASET] [GRAPH_ENCODER]
```

### 4. Cross-attention scores for adsorption site identification
The [`scripts/attn4sites.sh`](scripts/attn4sites.sh) is used to calculate the cross-attention scores of a trained AdsMT model on a GMAE dataset by running:
```bash
bash scripts/attn4sites.sh [CONFIG_PATH] [CHECKPOINT_PATH]
```
The output file will be stored at the `results_dir` in the log file.

We provide a notebook [`visualize/vis_3D.ipynb`](visualize/vis_3D.ipynb) to visualize and compare cross-attention score-colored surfaces with DFT-optimized adsorption configurations under GMAE.

## 🌈 Acknowledgements
This work was supported as part of NCCR Catalysis (grant number 180544), a National Centre of Competence in Research funded by the Swiss National Science Foundation.

This code repo is based on several existing repositories:
- [Open Catalyst Project](https://github.com/Open-Catalyst-Project/ocp)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [PyTorch](https://github.com/pytorch/pytorch)

## 📝 Citation
If you find our work useful, please consider citing it:
```bibtex
@article{adsmt,
author={Chen, Junwu and Huang, Xu and Hua, Cheng and He, Yulian and Schwaller, Philippe},
title={A multi-modal transformer for predicting global minimum adsorption energy},
journal={Nature Communications},
year={2025},
volume={16},
number={1},
pages={3232},
issn={2041-1723},
doi={10.1038/s41467-025-58499-7},
url={https://doi.org/10.1038/s41467-025-58499-7}
}
```

## 📫 Contact
If you have any question, welcome to contact me at:

Junwu Chen: junwu.chen@epfl.ch