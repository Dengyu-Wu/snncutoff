#!/bin/bash

python  ./scripts/evaluation.py \
        base.batch_size=128 \
        base.epochs=300 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'13152\' \
        base.data=\'cifar10\' \
        base.model=\'resnet18\' \
        base.dataset_path='datasets' \
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'qcfsconstrs\' \
        snn-train.regularizer=\'none\' \
        snn-train.multistep=False \
        snn-train.multistep_ann=False \
        snn-train.add_time_dim=True \
        snn-train.L=4 \
        snn-train.T=4 \
        snn-train.alpha=0.00 \
        \
        snn-test.epsilon=0.0 \
        snn-test.reset_mode='soft' \
        snn-test.model_path=\'outputs/cifar10-resnet18-ann1T4L-TETFalse-TEBNFalse-qcfsconstrs-none-alpha0.0-seed200-epochs300/cifar10.pth\' \
        