#!/bin/bash

python  ./snncutoff/scripts/training.py \
        base.epochs=120 \
        base.batch_size=512 \
        base.gpu_id=\'none\' \
        base.seed=1200 \
        base.port=\'13152\' \
        base.data=\'imagenet\' \
        base.model=\'vgg16\' \
        base.dataset_path='/users/wooooo/localscratch/datasets/LSVRC2012/' \
        base.checkpoint_save=True \
        base.checkpoint_path=none \
        \
        snn-train.method=\'ann\' \
        snn-train.ann_constrs=\'qcfsconstrs\' \
        snn-train.regularizer=\'none\' \
        snn-train.multistep=True \
        snn-train.add_time_dim=True \
        snn-train.L=16 \
        snn-train.T=1 \
        snn-train.alpha=0.00 