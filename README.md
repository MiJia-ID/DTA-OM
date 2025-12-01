# DTA-OM

A Pytorch Implementation of paper:

**An Artificial Intelligence Model for Accurate Drug-Target Affinity Prediction in Medicinal Chemistry**

Jia Mi, Jinghong Sun, JingLi, Chang Li, Jing Wan



Predicting Drug–Target Affinity (DTA) with high fidelity is critical for accelerating hit-to-lead optimization and understanding mechanism of action. While deep learning has transformed this field, current approaches often struggle with the effective encoding of protein semantics and the modeling of complex, non-covalent binding interactions. To address these limitations, we present a novel framework that synergizes evolutionary protein representations with multi-modal ligand profiling. On the target side, we employ Principal Component Analysis (PCA) to distill dense evolutionary information from ESM-2 embeddings, removing noise while retaining biologically relevant signals; this is fused with CNN-extracted local motifs to capture multi-scale features. On the ligand side, we ensure robust chemical space coverage by integrating molecular graphs with orthogonal descriptors—Morgan, Avalon, and MACCS keys—via an attention-guided fusion module. Furthermore, to mimic the dynamic nature of molecular recognition, we introduce a staged interaction mechanism combining cross- and self-attention to resolve fine-grained binding patterns. Extensive evaluations on benchmark datasets (Davis and KIBA) demonstrate that our model achieves state-of-the-art performance, particularly under stringent Novel-pair and Novel-drug settings. Crucially, the model's reliability is corroborated by molecular docking case studies, which validate the consistency between predicted affinities and structural interaction energies.

## 0. Overview of MTAF-DTA

![fig.1](fig.1.jpg)

Set up the environment:

In our experiment we use, Python 3.8.0 with PyTorch  2.0.1.

```
git clone https://github.com/MiJia-ID/DTA-OM.git
conda env create -f environment.yml
```

# 1. Dataset

The data should be in the format .csv: 'smiles', 'target_sequences', 'affinity'.

# 2. How to train

```
nohup python train.py 2>save_result/train&
```

# 3. To train YOUR model:

Your data should be in the format .csv, and the column names are: 'smiles', 'target_sequences', 'affinity'.

Generate the drug feature file from the given document using the [AMF.py](https://github.com/MiJia-ID/AMF.py)

Then you can freely tune the hyperparameter for your best performance.

# 4. To use OUR pre-trained model：
The trained model used in the manuscript can be downloaded from:
[Google Drive](https://drive.google.com/drive/folders/1rs7czPWuwOHWcGcsY54Dxht3ZwP0nR4O?usp=drive_link)
