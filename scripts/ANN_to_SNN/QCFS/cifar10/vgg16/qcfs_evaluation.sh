#!/bin/bash

python  ./scripts/evaluation.py \
        base.epochs=300 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'13152\' \
        base.data=\'cifar10\' \
        base.model=\'vgg16\' \
        base.dataset_path='datasets' \
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'qcfsconstrs\' \
        snn-train.regularizer=\'none\' \
        snn-train.multistep=False \
        snn-train.add_time_dim=True \
        snn-train.L=8 \
        snn-train.T=10 \
        snn-train.alpha=0.00 \
        \
        snn-test.sigma=1.0 \
        snn-test.reset_mode='soft' \
        snn-test.model_path=\'outputs/cifar10-vgg16-ann1T8L-qcfsconstrs-none-alpha0.0-seed1200-epochs300/cifar10-aideoserver.pth\'
        