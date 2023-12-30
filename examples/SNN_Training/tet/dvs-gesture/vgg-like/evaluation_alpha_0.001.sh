#!/bin/bash

python  ./snncutoff/scripts/evaluation.py \
        base.batch_size=128 \
        base.epochs=100 \
        base.gpu_id=\'1\' \
        base.seed=1200 \
        base.port=\'11152\' \
        base.data=\'cifar10-dvs\' \
        base.model=\'vggann\' \
        base.dataset_path=\'/LOCAL/dengyu/dvs_dataset/dvs-cifar10\'\
        \
        snn-train.method=\'snn\' \
        snn-train.ann_constrs=\'none\' \
        snn-train.snn_layers=\'baselayer\' \
        snn-train.regularizer=\'roe\' \
        snn-train.TET=False \
        snn-train.multistep=False \
        snn-train.add_time_dim=False \
        snn-train.T=10 \
        snn-train.alpha=0.0 \
        \
        snn-test.sigma=0.98 \
        snn-test.reset_mode='hard' \
        snn-test.model_path=\'outputs/rtmax-cifar10-dvs-vggann-snn10T4L-TETTrue-none-roe-alpha0.003-seed1200-epochs100/cifar10-dvs-aideoserver.pth\'
        # snn-test.model_path=\'outputs/rt-cifar10-dvs-vggann-snn10T4L-TETFalse-none-none-alpha0.0-seed1200-epochs100/cifar10-dvs-aideoserver.pth\'
        # snn-test.model_path=\'outputs/rt-cifar10-dvs-vggann-snn10T4L-none-roe-alpha0.003-seed1200-epochs100/cifar10-dvs-aideoserver.pth\'
        # snn-test.model_path=\'outputs/cifar10-dvs-vggann-snn10T4L-none-roe-alpha0.002-seed1200-epochs100/cifar10-dvs-aideoserver.pth\'
        