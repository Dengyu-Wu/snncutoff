# SNNCutoff
<p align="center">
<img src="./doc/pic/SNNCutoff.png" width="365">
</p>

Welcome to the SNNCutoff repository! This project is dedicated to advancing the capabilities of Spiking Neural Networks (SNNs) through innovative cutoff mechanisms and regularisation techniques. SNNCutoff aims to optimise SNNs for efficient and accurate performance, particularly focusing on both optimising training and inference stage for reliable classification.

## A New Metric
- **Optimal Cutoff Timestep (OCT)**: A optimal timestep that determines the minimal input processing duration for maintaining predictive reliability in SNNs. OCT is grounded in theoretical analysis and serves as a robust benchmark for assessing SNN models under different optimization algorithms.

## Cutoff Approximation 
- **Timestep (Baseline)**: Cutoff triggered using fixed timestep. 
- **Top-K**: Cutoff triggered using the gap between the top-1 and top-2 output predictions at each timestep. 


<!-- GETTING STARTED -->
## Getting Started
To begin using SNNCutoff, clone this repository and follow the setup instructions below. 
The package is tested in Python 3.9.10 and Pytorch 1.13.1.

### Installation

1. Clone the repo
```sh
git clone https://github.com/TACPSLab/easycutoff.git
```

2. Install Pytorch
```sh
pip pip install -r requirements. txt 
``` 

### Training Example
```sh
sh examples/ANN_to_SNN/qcfs/cifar10/vgg16/training.sh
```
or
```sh
sh examples/SNN_Training/tet/cifar10-dvs/vggsnn/training.sh
```

### Evaluation Example
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

This project originated from our research presented in the paper: [preprint](https://arxiv.org/abs/2301.09522). 

```
@article{wu2023optimising,
  title={Optimising Event-Driven Spiking Neural Network with Regularisation and Cutoff},
  author={Wu, Dengyu and Jin, Gaojie and Yu, Han and Yi, Xinping and Huang, Xiaowei},
  journal={arXiv preprint arXiv:2301.09522},
  year={2023}
}
```


## Acknowledgments
Special acknowledgment is given to the following projects for their influence on our coding style and structure:

- [OpenOOD](https://github.com/Jingkang50/OpenOOD) for its robust and versatile framework.
- [SNN-QCFS](https://github.com/putshua/SNN_conversion_QCFS), [SpKeras](https://github.com/Dengyu-Wu/spkeras)  for their innovative approaches in SNN conversion.
- [ChatGPT](https://chat.openai.com/auth/login) for the genatrative logo.

We extend our appreciation to all those who have contributed, directly or indirectly, to the success of this endeavor. Your contributions are deeply valued and have played a significant role in advancing the field of Spiking Neural Networks.




