#!/bin/bash

python  ./snncutoff/scripts/training.py \
        base.epochs=100 \
        base.gpu_id=\'0\' \
        base.seed=1200 \
        base.port=\'12322\' \
        base.data=\'cifar10-dvs\' \
        base.model=\'vggann\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10-1\'\
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'qcfsconstrs\' \
        snn-train.regularizer=\'roe\' \
        snn-train.multistep=True \
        snn-train.L=8 \
        snn-train.T=1 \
        snn-train.alpha=0.002