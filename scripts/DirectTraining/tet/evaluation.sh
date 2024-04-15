#!/bin/bash

python  ./scripts/evaluation.py \
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
        snn-train.ann_constrs=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'rcs\' \
        snn-train.TET=False \
        snn-train.multistep=False \
        snn-train.add_time_dim=False \
        snn-train.T=10 \
        snn-train.alpha=0.0 \
        \
        snn-test.epsilon=1.0 \
        snn-test.reset_mode='hard' \
        snn-test.model_path=\'/LOCAL2/dengyu/MySNN/easycutoff/outputs/cifar10-dvs.pth\' \