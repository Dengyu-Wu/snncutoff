# Easy Cutoff


<!-- GETTING STARTED -->
## Getting Started
The package is tested in Python 3.9.10 and Pytorch 1.13.1.

### Prerequisites

1. Install Pytorch
```sh
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

### Installation

1. Clone the repo
```sh
git clone https://github.com/TACPSLab/easycutoff.git
```

### Training Example
```sh
sh examples/ANN_to_SNN/qcfs/cifar10/vgg16/training.sh
```
or
```sh
sh examples/SNN_Training/tet/cifar10-dvs/vggsnn/training.sh
```

### Test Example
```sh
python examples/SNN_Training/qcfs/cifar10-dvs/vggsnn/evaluation.sh
```
or
```sh
python examples/SNN_Training/tet/cifar10-dvs/vggsnn/evaluation.sh
```
