#!/bin/bash

python  ./snncutoff/scripts/evaluation.py \
        base.batch_size=32 \
        base.epochs=100 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'12112\' \
        base.data=\'dvs128-gesture\' \
        base.model=\'vgg-gesture\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-gesture\'\
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'roe\' \
        snn-train.TET=True \
        snn-train.multistep=True \
        snn-train.T=16 \
        snn-train.alpha=0.002