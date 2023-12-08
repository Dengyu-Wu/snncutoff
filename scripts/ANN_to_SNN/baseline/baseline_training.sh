#!/bin/bash

python /LOCAL2/dengyu/MySNN/easycutoff/scripts/training.py \
        base.epochs=300 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'14152\' \
        base.data=\'cifar10-dvs\' \
        base.model=\'vggann\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10-1\'\
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'baseconstrs\' \
        snn-train.regularizer=\'none\' \
        snn-train.multistep=False \
        snn-train.T=10 \
        snn-train.alpha=0.00 