#!/bin/bash

python  ./scripts/training.py \
        base.batch_size=128 \
        base.epochs=300 \
        base.gpu_id=\'0\' \
        base.seed=1200 \
        base.port=\'11152\' \
        base.data=\'cifar10\' \
        base.model=\'sew_resnet18\' \
        base.dataset_path=\'datasets\' \
        base.checkpoint_save=False \
        base.checkpoint_path=none \
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'none\' \
        snn-train.arch_conversion=False \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'rcs\' \
        snn-train.loss='tet' \
        snn-train.multistep=True \
        snn-train.add_time_dim=True \
        snn-train.T=2 \
        snn-train.alpha=0.00 \
        \
        neuron_params.vthr=1.0 \
        neuron_params.tau=0.5 \
        neuron_params.mem_init=0.0 \
        neuron_params.multistep=True \
        neuron_params.reset_mode='hard' 