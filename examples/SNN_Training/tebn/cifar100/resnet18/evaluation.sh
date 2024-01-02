#!/bin/bash

python  ./snncutoff/scripts/evaluation.py \
        base.batch_size=128 \
        base.epochs=300 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'14152\' \
        base.data=\'cifar100\' \
        base.model=\'resnet18\' \
        base.dataset_path=\'datasets\' \
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'baseconstrs\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'roe\' \
        snn-train.TET=False \
        snn-train.TBN=True \
        snn-train.multistep=False \
        snn-train.add_time_dim=True \
        snn-train.T=6 \
        snn-train.alpha=0.00 \
        \
        snn-test.sigma=1.0 \
        snn-test.reset_mode='hard' \
        snn-test.model_path=\'outputs/cifar100-resnet18-snn6T4L-TETFalse-TBNTrue-baseconstrs-roe-alpha0.002-seed1200-epochs300/cifar100-aideoserver.pth\'
        