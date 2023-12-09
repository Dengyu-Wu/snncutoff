#!/bin/bash

python ./scripts/training.py \
        base.epochs=300 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'14122\' \
        base.data=\'cifar10-dvs\' \
        base.model=\'vggann\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10-1\'\
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'qcfsconstrs\' \
        snn-train.regularizer=\'roe\' \
        snn-train.multistep=True \
        snn-train.T=8 \
        snn-train.alpha=0.002