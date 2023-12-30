#!/bin/bash

python  ./scripts/training.py \
        base.batch_size=128 \
        base.epochs=300 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'12152\' \
        base.data=\'cifar100\' \
        base.model=\'resnet18\' \
        base.dataset_path=\'datasets\' \
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'qcfsconstrs\' \
        snn-train.regularizer=\'roe\' \
        snn-train.TET=False \
        snn-train.TBN=False \
        snn-train.multistep=True \
        snn-train.add_time_dim=True \
        snn-train.T=1 \
        snn-train.L=8 \
        snn-train.alpha=0.001