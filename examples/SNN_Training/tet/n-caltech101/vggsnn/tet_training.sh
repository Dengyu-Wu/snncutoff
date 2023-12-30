#!/bin/bash

python  ./scripts/training.py \
        base.batch_size=128 \
        base.epochs=100 \
        base.gpu_id=\'0\' \
        base.seed=1200 \
        base.port=\'11112\' \
        base.data=\'ncaltech101\' \
        base.model=\'vggann-ncaltech101\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10\'\
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'none\' \
        snn-train.TET=True \
        snn-train.multistep=True \
        snn-train.T=10 \
        snn-train.alpha=0.00