#!/bin/bash

python  ./scripts/training.py \
        base.epochs=300 \
        base.gpu_id=\'0\' \
        base.seed=1200 \
        base.port=\'14152\' \
        base.data=\'cifar10\' \
        base.model=\'vgg16\' \
        base.dataset_path=\'./datasets'\
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'baseconstrs\' \
        snn-train.regularizer=\'none\' \
        snn-train.multistep=True \
        snn-train.T=1 \
        snn-train.alpha=0.00 