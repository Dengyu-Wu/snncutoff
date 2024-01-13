# SNNCutoff
<p align="center">
<img src="https://github.com/Dengyu-Wu/snncutoff/raw/main/docs/_static/logo.png" width="700">
</p>

Welcome to the SNNCutoff! This repository is centered on advancing the training and evaluation of Spiking Neural Networks (SNNs) with an eye towards innovative cutoff mechanisms. SNNCutoff aims to refine the efficiency of SNNs, ensuring robust computing that does not compromise on performance accuracy. It is particularly dedicated to optimizing both the training and inference phases to secure reliable classification outcomes.

## Framework 

SNNCutoff provides a training workflow leveraging PyTorch for model conversion. It enables the seamless transformation of Artificial Neural Network (ANN) architectures with ReLU activation into intermediate models. These models can be tailored with specific constraints optimising ANN training for SNN, as well as equipped with spiking layers for direct SNN training optimisation.

<p align="center">
<img src="https://github.com/Dengyu-Wu/snncutoff/raw/main/docs/_static/framework.png" width="800">
</p>

## A New Metric
- **Optimal Cutoff Timestep (OCT)**: A optimal timestep that determines the minimal input processing duration for maintaining predictive reliability in SNNs. OCT is grounded in theoretical analysis and serves as a robust benchmark for assessing SNN models under different optimization algorithms.

## Cutoff Approximation 
- **Timestep (Baseline)**: Cutoff triggered using fixed timestep. 
- **Top-K**: Cutoff triggered using the gap between the top-1 and top-2 output predictions at each timestep. 
- **Others**: Coming soon. 


<!-- GETTING STARTED -->
## Getting Started
To begin using SNNCutoff, clone this repository and follow the setup instructions below. 

### Installation

1. Clone the repo
```sh
git clone https://github.com/TACPSLab/easycutoff.git
```

2. Install Pytorch
```sh
pip pip install -r requirements. txt 
``` 

### Training and Evaluation 
We provide training and evaluation scripts in [examples](/examples). 

## Contributing

We welcome contributions from the community! If you have suggestions for improvement or want to contribute to the codebase, please read our [Contributing Guidelines](./contributing.md) for more information on submitting pull requests.


## Acknowledgments
Special acknowledgment is given to the following projects for their influence on our coding style and structure:

- [OpenOOD](https://github.com/Jingkang50/OpenOOD) for its robust and versatile framework.
- [SNN-QCFS](https://github.com/putshua/SNN_conversion_QCFS), [SpKeras](https://github.com/Dengyu-Wu/spkeras)  for their innovative approaches in SNN conversion.
- [ChatGPT](https://chat.openai.com/auth/login) for the generative logo.

We extend our appreciation to all those who have contributed, directly or indirectly, to the success of this endeavor. Your contributions are deeply valued and have played a significant role in advancing the field of Spiking Neural Networks.




