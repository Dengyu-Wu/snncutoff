#!/bin/bash

python  ./snncutoff/scripts/evaluation.py \
        base.batch_size=128 \
        base.epochs=300 \
        base.gpu_id=\'2\' \
        base.seed=1200 \
        base.port=\'15452\' \
        base.data=\'cifar100\' \
        base.model=\'resnet18\' \
        base.dataset_path=\'datasets\' \
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'baseconstrs\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'roe\' \
        snn-train.TET=True \
        snn-train.multistep=True \
        snn-train.add_time_dim=True \
        snn-train.T=6 \
        snn-train.alpha=0.002