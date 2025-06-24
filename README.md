# Compositional Targeted Multi-Label Universal Perturbations 
*CVPR 2025 · Hassan Mahmood · Ehsan Elhamifar* 
[[**Paper PDF**]](https://openaccess.thecvf.com/content/CVPR2025/papers/Mahmood_Compositional_Targeted_Multi-Label_Universal_Perturbations_CVPR_2025_paper.pdf)

## Introduction

> We propose a *compositional* framework to generate universal adversarial perturbations for multi-label learning models. We show that a simple independence assumption on label-wise universal perturbations naturally leads to an efficient optimization that requires learning **affine convex cones** spanned by label-wise universal perturbations, significantly reducing the problem complexity to **linear** time and space. During inference, the framework allows generating universal perturbations for novel combinations of classes in **constant time**. We evaluate the performance of our method on NUS-WIDE, MS-COCO, and OpenImages using state-of-the-art multi-label recognition models. 

## Overview
We provide the code to 
1. Train/learn compositional universal adversarial perturbations using $CMLU_\beta$.
2. Evaluate the learnt universal adversarial perturbations.

**Note**: I have provided the code for evaluation and training $CMLU_\beta$. I will add the code for training baselines and visualization soon.

## Environment Setup
Using conda, create a new environment and install the dependencies:
```
conda env create -f environment.yml
conda activate asl1
```

## Config File
The config files in directory **configs/** are used to set the variables for the experiments. Each file would contain some `local` experiment-specific variables and some `global` variables.

The config files are dividied into three main parts:
1. Experiment blocks (`[<experiment_name>]`)
2. Metadata (`[metadata]`)
3. Global variables (`[globalvars]`)

### Parameter Reference
#### Experiment-specific

- **method_name**: Alias of the attack or algorithm to run. It can be of the following:\
`oracle`, `or-s`, `or-c`, `nag`, `cmlu_a`, `cmlu_b`
    If the method_name is not one of those, the code will raise an exception.
- **targetsize**: Combination size of target classes $|\Omega|$.
- **nontargetlossscale**: Weighting factor for non-target loss term. Used for training. 
- **eps_norm**: Perturbation budget (e.g. $\epsilon = 0.05$).
- **p_norm**: Norm type (inf, 2, etc.).
- **checkpoint_load_path**: Path to the store universals $U$. Used for evaluation.

#### MetaData
- **classnames_path**: Text file with one class name per line.
- **main_model_path**: Path to the primary model used for predictions.
- **target_classes_dir**: For a dataset with classes $C$, we can have ${|C|\choose k}$ possible combinations of classes of size $k$. Since this is exponential, we choose only a subset of those ${|C|\choose k}$. This directory contains pickle files for different $k$ and in each of those files, we have a subset/list of combinations of classes $C$ of size $k$.

#### Global Variables
- **weights_dir, log_folder, stats_folder**: Directories to store weights and output logs/stats.
- **image_size, batch_size, num_epochs, step_size**: Standard training settings. 
- **training_device**: e.g. cuda:0 or cpu.
- **model_name, dataset_name**: Short identifiers used in naming. model_name can be `asl` or `mldecoder`. dataset_name can be `NUSWIDE` or `MSCOCO`.

- **Data Paths**
    * **\*_images_dir**: Path to train/val images directory.
    * **\*_labels_file**: .npy file contains the clean labels for the dataset
    * **\*_img_ids_file**: pickle file contains the paths to images for the dataset
    * **\*_pred_file**: .npy file contains the prediction of the model on clean images.
    * **\*_indices_dir**:  For a given combination of classes of size $k$, we choose the images from train/val set that contain those classes. For example, for $k = 3$, assume we have {Person, Car, Cat}. We choose all train/val images that the model predicts to contain those classes and store the indices of those images in this directory.

### Creating New Experiments
To create a new experiment,
1. Copy one of the existing `[asl_cmlu_b_targetX]` blocks.
2. Rename the header (experiment name).
3. Adjust the parameters that need changing (e.g. targetsize, stepsize).

When launching your script, use the experiment name.

## Data Preparation 

Here are the instructions for NUS-WIDE and MS-COCO

### Download Images
We use the data provided by the authors of ASL paper.
1. **NUS-WIDE**: Download the dataset from [ASL-Git](https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md) and extract the folder contents.
2. **MSCOCO**: Similar to [ASL](https://github.com/Alibaba-MIIL/ASL/blob/main/Datasets.md), we use images from official repository and 2014 Train/Val annotations.


### Download Labels, Model Predictions, MetaData, Checkpoints
You can download the clean labels, predictions on clean images by ASL and MLDecoder models, and the metadata for each dataset from this [Link](https://drive.google.com/drive/folders/1WjM4d9pX4oRHMkyCSsicOVJBrisr8w-q?usp=sharing).

In the main code folder (at root directory MLLUniversals/), extract the contents of each tar file as:
```
tar xzf [FILE NAME]
```

We have provided the checkpoints for ASL on NUS-WIDE and MSCOCO dataset.

## Model Preparation 
You can use any multi-label learning model. In this repository, we provide code for ASL-based TResNet-L model. \
Download the model weights from the official repo of ASL: [model weights](https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md) and set the path variable `main_model_path` in nus_asl and mscoco_asl config files.




## Training/Learning Compositional Universal Adversarial Perturbations
After preparing the data, model, and setting the variables in config files, run the following command to start training:

```
python trainscripts/train_cmlu_b.py --configfile [CONFIG FILE PATH] --mode train [EXPERIMENT NAME]
```
For example, to train $CMLU_\beta$ on NUS-WIDE dataset and ASL model,
```
python trainscripts/train_cmlu_b.py --configfile configs/nus_asl.ini --mode train asl_cmlu_b_target1
```

## Evaluation
To evaluate:

```
python testscripts/evaluate_universals.py --configfile [CONFIG FILE PATH] --mode test [EXPERIMENT NAME]
```
For example, to evaluate $CMLU_\beta$ using target size 5:
```
python testscripts/evaluate_universals.py --configfile configs/nus_asl.ini --mode test asl_cmlu_b_target5
```

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{Mahmood:CVPR25,
  title={Compositional Targeted Multi-Label Universal Perturbations},
  author={Mahmood, Hassan and Elhamifar, Ehsan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20580--20591},
  year={2025}
}
```

## Contact

For questions or issues, please contact:
- **Hassan Mahmood**: [mahmood.h@northeastern.edu](mailto:mahmood.h@northeastern.edu)