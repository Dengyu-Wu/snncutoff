# Easy Cutoff


<!-- GETTING STARTED -->
## Getting Started
The package is tested in Python 3.9.10 and Pytorch 1.13.1.

### Prerequisites

1. Install tensorflow
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
python scripts/training.py base.gpu_id=\'0\
```

### Test Example
```sh
python scripts/test.py base.gpu_id=\'0\ snn.mode=cutoff
```
