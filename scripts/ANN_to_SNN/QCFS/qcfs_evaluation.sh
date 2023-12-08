#!/bin/bash

python /LOCAL2/dengyu/MySNN/easycutoff/scripts/training.py \
        base.gpu_id=\'0\' \
        base.seed=2000 \
        base.port=\'13352\' \
        base.dataset=\'cifar10-dvs\' \
        base.model=\'vggann\' \
        \
        snn-train.method='ann' \
        snn-train.alpha=0.002 \         
