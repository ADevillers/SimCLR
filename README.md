# PyTorch SimCLR for HPC Clusters with Slurm

## Overview

This project provides our PyTorch reimplementation of SimCLR. It is specifically designed to leverage high-performance computing (HPC) clusters using Slurm workload manager, and it comes ready for use on the Jean Zay cluster. It includes full training and linear evaluation pipelines, with support for CIFAR10 and ImageNet datasets.

For an in-depth analysis of the reimplementation process and results, please refer to the replication report provided in `report.pdf`.

## Results

We've successfully replicated and even improved upon the original SimCLR results. Details of these results, which include Top-1 and Top-5 accuracies on CIFAR10 and ImageNet datasets, are provided in the table below. The trained models weights are available under the `checkpoints` folder for further inspection or use.

### CIFAR10
| Implementation                  | Top-1 Accuracy | Top-5 Accuracy |
|---------------------------------|----------------|----------------|
| Original                        | ~91.0%         | Not Reported   |
| Our Replication                 | 91.70%         | 99.71%         |
| Our Replication (3-layer head)  | 91.93%         | 99.79%         |

### ImageNet
| Implementation                  | Top-1 Accuracy | Top-5 Accuracy |
|---------------------------------|----------------|----------------|
| Original                        | 69.3%          | 89.0%          |
| Our Replication                 | 70.78%         | 90.36%         |
| Our Replication (3-layer head)  | 71.84%         | 90.53%         |

## Requirements

Before running the code, please install the necessary packages by running:

```bash
pip install -r requirements.txt
```

## Usage

The project is tailored for the Jean Zay cluster with ready-to-use Slurm scripts. Non-Jean Zay Slurm users can adjust the scripts to match their environment's configurations by modifying partition names, module loads, and other SBATCH directives.

For those not on Slurm or on systems with different specifications, the Python commands can be run directly from the command line. Note that due to the intensive computational requirements of these experimentations, you may need to alter the hyperparameters like batch size to accommodate to your system's capabilities, wich may lower performances.

### Arguments

The Python scripts (`main.py` and `eval.py`) accept several command-line arguments to tailor the execution to your hardware and experimental setup. Below is an explanation of some key arguments:

- --computer: Specifies the computing environment ('jeanzay' or 'other').
- --hardware: Defines the hardware setup ('cpu', 'mono-gpu', 'multi-gpu').
- --precision: Sets the numerical precision ('mixed' or 'normal').
- --nb_workers: The number of worker threads for data loading.
- --expe_name: A name for the experiment, used in saving models.
- --dataset_name: The dataset to use ('cifar10' or 'imagenet').
- --resnet_type: The type of ResNet model ('resnet18', 'resnet50', etc.).
- --nb_epochs: The total number of epochs to train for.

For a comprehensive list of arguments and their descriptions, consult the source code.

### Training

#### CIFAR10

```bash
python src/main.py --computer=other --hardware=mono-gpu --precision=mixed --nb_workers=10 --expe_name=simclr_cifar10 --dataset_name=cifar10 --resnet_type=resnet18 --nb_epochs=800 --nb_epochs_warmup=10 --batch_size=512 --lr_init=4.0 --momentum=0.9 --weight_decay=1e-6 --eta=1e-3 --z_dim=128 --temperature_z=0.5 --clsf_every=100 --save_every=100 --nb_epochs_clsf=90 --batch_size_clsf=256 --lr_init_clsf=0.2 --momentum_clsf=0.9 --weight_decay_clsf=0.0
```

#### ImageNet

```bash
python src/main.py --computer=other --hardware=mono-gpu --precision=mixed --nb_workers=10 --expe_name=simclr_imagenet --dataset_name=imagenet --resnet_type=resnet50 --nb_epochs=800 --nb_epochs_warmup=10 --batch_size=4096 --lr_init=4.8 --momentum=0.9 --weight_decay=1e-6 --eta=1e-3 --z_dim=128 --temperature_z=0.2 --clsf_every=100 --save_every=100 --nb_epochs_clsf=90 --batch_size_clsf=256 --lr_init_clsf=0.2 --momentum_clsf=0.9 --weight_decay_clsf=0.0
```

### Evaluation

#### CIFAR10

```bash
python src/eval.py --computer=other --hardware=mono-gpu --precision=mixed --nb_workers=10 --dataset_name=cifar10 --resnet_type=resnet18 --z_dim=128 --nb_epochs_clsf=90 --batch_size_clsf=256 --lr_init_clsf=0.2 --momentum_clsf=0.9 --weight_decay_clsf=0.0 --checkpoint=./checkpoints/weights_simclr_cifar10.pt
```

#### ImageNet

```bash
python src/eval.py --computer=other --hardware=mono-gpu --precision=mixed --nb_workers=10 --dataset_name=imagenet --resnet_type=resnet50 --z_dim=128 --nb_epochs_clsf=90 --batch_size_clsf=256 --lr_init_clsf=0.2 --momentum_clsf=0.9 --weight_decay_clsf=0.0 --checkpoint=./checkpoints/weights_simclr_imagenet.pt
```
