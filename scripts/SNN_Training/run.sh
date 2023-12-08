#!/bin/bash

# nohup python /LOCAL2/dengyu/MySNN/easycutoff/scripts/training.py base.gpu_id=\'0\' base.seed=2000 snn-train.alpha=0.001 base.port=\'13252\' hydra.run.dir=./output&
# sleep 1
nohup python /LOCAL2/dengyu/MySNN/easycutoff/scripts/training.py base.gpu_id=\'0\' base.seed=2000 snn-train.alpha=0.0001 base.port=\'13352\' &
sleep 1
# nohup python /LOCAL2/dengyu/MySNN/easycutoff/scripts/training.py base.gpu_id=\'1\' base.seed=2000 snn-train.alpha=0.0002 base.port=\'13452\' &
sleep 1
# nohup python /LOCAL2/dengyu/MySNN/easycutoff/scripts/training.py base.gpu_id=\'1\' base.seed=2000 snn-train.alpha=0.002 base.port=\'13552\' &