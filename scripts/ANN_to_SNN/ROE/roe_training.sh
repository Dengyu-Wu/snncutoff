#!/bin/bash

python /LOCAL2/dengyu/MySNN/easycutoff/scripts/training.py \
        base.epochs=300 \
        base.gpu_id=\'0\' \
        base.seed=1200 \
        base.port=\'13352\' \
        base.data=\'cifar10-dvs\' \
        base.model=\'vggann\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10-1\'\
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'baseconstrs\' \
        snn-train.regularizer=\'roe\' \
        snn-train.multistep=False \
        snn-train.T=4 \
        snn-train.alpha=0.002 
