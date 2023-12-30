#!/bin/bash

python  ./snncutoff/scripts/evaluation.py \
        base.batch_size=32 \
        base.epochs=100 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'11112\' \
        base.data=\'dvs128-gesture\' \
        base.model=\'vgg-gesture\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-gesture\'\
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'roe\' \
        snn-train.TET=False \
        snn-train.multistep=False \
        snn-train.add_time_dim=False \
        snn-train.T=16 \
        snn-train.alpha=0.0 \
        \
        snn-test.sigma=1.0 \
        snn-test.reset_mode='hard' \
        snn-test.model_path=\'outputs/dvs128-gesture-vgg-gesture-snn16T4L-TETTrue-TBNFalse-none-none-alpha0.005-seed1200-epochs100/dvs128-gesture-aideoserver.pth\'
        