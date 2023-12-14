#!/bin/bash

python  ./scripts/evaluation.py \
        base.gpu_id=\'0\' \
        base.seed=2000 \
        base.port=\'13352\' \
        base.data=\'cifar10\' \
        base.model=\'resnet18\' \
        base.dataset_path='datasets' \
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'baseconstrs\' \
        snn-train.regularizer=\'none\' \
        snn-train.multistep=True \
        snn-train.T=4 \
        snn-train.alpha=0.00 \
        snn-train.add_time_dim=True \
        \
        snn-test.sigma=1.0 \
        snn-test.reset_mode='soft' \
        snn-test.model_path=\'/LOCAL2/dengyu/MySNN/easycutoff/outputs/cifar10-vgg16-ann1T4L-baseconstrs-none-alpha0.0-seed1200-epochs300/cifar10-aideoserver.pth\'\
        