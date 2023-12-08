import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from easycutoff.models.resnet_models import resnet19
from easycutoff.models.vggsnns import *
from easycutoff import data_loaders
from easycutoff.functions import TET_loss
import numpy as np
from configs import *
from omegaconf import DictConfig, OmegaConf
import hydra
from easycutoff.utils import replace_activation_by_neuron, ann_to_snn_conversion, reset_neuron
from easycutoff.Evaluator import Evaluator
from easycutoff.utils import multi_to_single_step,add_ann_constraints

@hydra.main(version_base=None, config_path='../configs', config_name='test')
def main(cfg: DictConfig):
    all_conf = TestConfig(**cfg['base'],**cfg['snn-test'])
    # all_conf.nprocs = torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = all_conf.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = data_loaders.get_data_loaders(path=all_conf.dataset_path, data=all_conf.data, resize=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=all_conf.batch_size,
                                              shuffle=False, num_workers=all_conf.workers, pin_memory=True)
    models = [VGGANN() for i in range(1)]  
    from easycutoff.ann_constrs import QCFSConstrs
    from easycutoff.regularizer import ROE
    models[0] = add_ann_constraints(models[0], 4, ann_constrs=QCFSConstrs, regularizer=ROE())
    i= 0
    path = all_conf.model_path
    dataset='Cifar10DVS/'
    # alpha = all_conf.alpha
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    # state_dict = torch.load(path+'saved_models/'+dataset+'alpha_'+str(alpha)+'_seed_'+str(one_seed)+'.pth', map_location=torch.device('cpu'))  
    models[i].load_state_dict(state_dict, strict=False)

    # print(models[i])
    layers = []
    # summary(models[i])

    # models,layers = ann_to_snn_conversion(models[i],layers)
    # models = nn.Sequential(*list(layers))
    # models = models[i]
    models[i] = multi_to_single_step(models[i])
    # models = torch.nn.DataParallel(models[i])
    # models.to(device)  
    print(models[i])  
    models[i].eval()
    models[i].cuda()
    evaluator = Evaluator(models[0])
    acc, loss = evaluator.evaluation(test_loader)
    print(acc)
    print(loss)
    acc, loss = evaluator.aoi_evaluation(test_loader)
    print(acc)
    print(loss)
if __name__ == '__main__':
   main()
