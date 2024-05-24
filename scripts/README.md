# SNNCutoff Scripts

The scripts of `training.py` and `evaluation.py` accept various parameters to control different aspects of the training and evaluation.


## Content

- Conversion: examples based on conversion algorithm, e.g., QCFS
- DirectTraining: examples based on direction training, e.g., TET and TEBN

### Bash Script Example

```bash
#!/bin/bash

python ./scripts/training.py \
        base.batch_size=128 \
        base.epochs=300 \
        base.gpu_id='1' \
        base.seed=1200 \
        base.port='11152' \
        base.data='cifar10' \
        base.model='sew_resnet18' \
        base.dataset_path='datasets' \
        \
        snn-train.method='snn' \
        snn-train.arch_conversion=False \
        snn-train.ann_constrs='baseconstrs' \
        snn-train.snn_layers='simplebaselayer' \
        snn-train.regularizer='rcs' \
        snn-train.TET=True \
        snn-train.multistep=True \
        snn-train.add_time_dim=True \
        snn-train.T=6 \
        snn-train.alpha=0.00
```

#### Base Parameters

- `base.batch_size`: Specifies the batch size for training. Default is `128`.
- `base.epochs`: Defines the number of epochs for training. Default is `300`.
- `base.gpu_id`: Indicates the GPU ID to be used for training. Default is `'1'`.
- `base.seed`: Sets the random seed for reproducibility. Default is `1200`.
- `base.port`: Port number for distributed training. Default is `'11152'`.
- `base.data`: Dataset to be used. For example, `cifar10`.
- `base.model`: Model architecture to be used. For example, `sew_resnet18`, `vgg`
- `base.dataset_path`: Path to the dataset directory. Default is `'datasets'`.

#### SNN Training Parameters

- `snn-train.method`: Training method. For example, `'snn'`.
- `snn-train.arch_conversion`: Boolean to enable or disable architecture conversion. Default is `False`.
- `snn-train.ann_constrs`: ANN constraints. For example, `'baseconstrs'`.
- `snn-train.snn_layers`: Specifies the SNN layers. For example, `'simplebaselayer'`.
- `snn-train.regularizer`: Regularizer to be used. For example, `'rcs'`.
- `snn-train.TET`: Boolean to enable or disable TET. Default is `True`.
- `snn-train.multistep`: Boolean to enable or disable multi-step training. Default is `True`.
- `snn-train.add_time_dim`: Boolean to add a time dimension. Default is `True`.
- `snn-train.T`: Time steps for SNN training. Default is `6`.
- `snn-train.alpha`: Alpha parameter for the SNN training. Default is `0.00`.

### Training and Evaluation

```sh
sh scripts/DirectTraining/tet/training.sh
```
or 
```sh
sh scripts/DirectTraining/tet/evaluation.sh
```
