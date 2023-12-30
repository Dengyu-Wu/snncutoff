#!/bin/bash

python  ./scripts/training.py \
        base.batch_size=32 \
        base.epochs=100 \
        base.gpu_id=\'0\' \
        base.seed=1200 \
        base.port=\'11112\' \
        base.data=\'dvs128-gesture\' \
        base.model=\'vgg-gesture\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-gesture\'\
        base.checkpoint_save=True\
        base.checkpoint_path=\'outputs/dvs128-gesture-vgg-gesture-snn16T4L-TETTrue-TBNFalse-none-none-alpha0.005-seed1200-epochs100/dvs128-gesture-aideoserver_checkpoint.pth\'\
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'none\' \
        snn-train.TET=True \
        snn-train.multistep=True \
        snn-train.T=16 \
        snn-train.alpha=0.005