#!/bin/bash

python  ./scripts/training.py \
        base.batch_size=128 \
        base.epochs=300 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'11152\' \
        base.data=\'cifar10\' \
        base.model=\'sew_resnet18\' \
        base.dataset_path=\'datasets\' \
        \
        snn-train.method=\'snn\' \
        snn-train.arch_conversion=False \
        snn-train.ann_constrs=\'baseconstrs\' \
        snn-train.snn_layers=\'simplebaselayer\' \
        snn-train.regularizer=\'rcs\' \
        snn-train.TET=True \
        snn-train.multistep=True \
        snn-train.add_time_dim=True \
        snn-train.T=6 \
        snn-train.alpha=0.00