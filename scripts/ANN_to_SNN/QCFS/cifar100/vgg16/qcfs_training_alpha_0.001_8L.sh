#!/bin/bash

python  ./scripts/training.py \
        base.epochs=300 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'11051\' \
        base.data=\'cifar10\' \
        base.model=\'vgg16\' \
        base.dataset_path='datasets' \
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'qcfsconstrs\' \
        snn-train.regularizer=\'roe\' \
        snn-train.multistep=True \
        snn-train.add_time_dim=True \
        snn-train.L=8 \
        snn-train.T=1 \
        snn-train.alpha=0.003