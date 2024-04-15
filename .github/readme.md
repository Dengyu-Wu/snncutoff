<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../docs/_static/logo_dark.svg" width="700">
    <img alt="Text changing depending on mode. Light: 'So light!' Dark: 'So dark!'" src="../docs/_static/logo_light.svg"  width="700">
  </picture>
</div>

#

**SNNCutoff** is a Python package developed with a PyTorch backend, designed primarily for evaluating Spiking Neural Networks (SNNs). It offers:

- **SNN Evaluation**:
  - Utilizing detailed performance metrics, e.g., accuracy, latency and operations.
  - Capabilities for conducting adaptive inference or cutoff of SNNs.

- **SNN Training**:
  - While the emphasis is on evaluation, the toolkit also supports a diverse array of training algorithms.

# Overview 

<p align="center">
<img src="../docs/_static/framework.png" width="800">
</p>

- **A New Metric**:
  - **Optimal Cutoff Timestep (OCT)**: A optimal timestep that determines the minimal input processing duration for maintaining predictive reliability in SNNs. OCT is grounded in theoretical analysis and serves as a robust benchmark for assessing SNN models under different optimization algorithms.

- **Cutoff Approximation**:
  - **Timestep (Baseline)**: Cutoff triggered using fixed timestep. 
  - **Top-K**: Cutoff triggered using the gap between the top-1 and top-2 output predictions at each timestep. 
  - **Others**: Coming soon. 

More details in [Documentation](https://dengyu-wu.github.io/snncutoff/).

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
pip install -r requirements. txt 
``` 

### Training and Evaluation 
We provide training and evaluation examples in [scripts](/scripts). 

## Contributing

Check the [contributing guidelines](./contributing.md) if you want to get involved with developing SNNCutoff.

## Acknowledgments

We extend our appreciation to everyone who has contributed to the development of this project, both directly and indirectly.


