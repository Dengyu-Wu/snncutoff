# SNNCutoff: Optimising Spiking Neural Networks with Regularisation and Cutoff 

Welcome to the SNNCutoff repository! This project is dedicated to advancing the capabilities of Spiking Neural Networks (SNNs) through innovative cutoff mechanisms and regularisation techniques. SNNCutoff aims to optimise SNNs for efficient and accurate performance, particularly focusing on both optimising training and inference stage for reliable classification.

## Key Features

- **Optimal Cutoff Technique (OCT)**: A new metric that determines the minimal input processing duration for maintaining predictive reliability in SNNs. OCT is grounded in theoretical analysis and serves as a robust benchmark for assessing SNN models under different optimization algorithms.

- **Regularization Methodology**: Our custom-designed regularizer influences activation distributions during ANN or SNN training. It addresses 'worst-case' input scenarios that typically lead to early-timestep inference failures, enhancing overall network robustness.

- **Adaptable Framework**: Compatible with various SNN training methods, including Quantised Clip-Flip-Shift (QCFS) and Temporal Efficient Training (TET). The framework is versatile across different datasets and network architectures, with a focus on both frame-based and event-based inputs.

- **Comprehensive Evaluation**: Extensive testing across multiple standard datasets and network models to demonstrate the effectiveness of our approach in enhancing SNN performance.

## Getting Started

To begin using SNNCutoff, clone this repository and follow the setup instructions provided in [Installation.md](/Installation.md). Detailed documentation can be found in [Documentation.md](/Documentation.md), which includes guidelines for training, evaluation, and customization.

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

## Contributing

We welcome contributions from the community! If you have suggestions for improvement or want to contribute to the codebase, please read our [Contributing Guidelines](/CONTRIBUTING.md) for more information on submitting pull requests.

## License

This project is licensed under the [MIT License](/LICENSE) - see the license file for details.

## Research Paper

This project is supported by our research presented in the paper: [preprint](https://arxiv.org/abs/2301.09522). The methodologies and findings outlined in our paper provide the foundational framework for the SNNCutoff project.


## Acknowledgments
Special acknowledgment is given to the following projects for their influence on our coding style and structure:

- [OpenOOD](https://github.com/Jingkang50/OpenOOD) for its robust and versatile framework.
- [SNN-QCFS](https://github.com/putshua/SNN_conversion_QCFS), [SpKeras](https://github.com/Dengyu-Wu/spkeras)  for their innovative approaches in SNN conversion.

We extend our appreciation to all those who have contributed, directly or indirectly, to the success of this endeavor. Your contributions are deeply valued and have played a significant role in advancing the field of Spiking Neural Networks.




