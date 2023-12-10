#!/bin/bash

python  ./scripts/training.py \
        base.epochs=300 \
        base.gpu_id=\'2\' \
        base.seed=1200 \
        base.port=\'14152\' \
        base.data=\'cifar10-dvs\' \
        base.model=\'vggsnn\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10\'\
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'baseconstrs\' \
        snn-train.regularizer=\'none\' \
        snn-train.multistep=True \
        snn-train.T=10 \
        snn-train.alpha=0.00 