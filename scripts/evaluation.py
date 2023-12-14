import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from snncutoff.models.vggsnns import *
from snncutoff import data_loaders
import numpy as np
from configs import *
from omegaconf import DictConfig, OmegaConf
import hydra
from snncutoff.Evaluator import Evaluator
from snncutoff.utils import multi_to_single_step,add_ann_constraints
from snncutoff import get_models 
from snncutoff.snncase import SNNCASE

@hydra.main(version_base=None, config_path='../configs', config_name='test')
def main(cfg: DictConfig):
    args = TestConfig(**cfg['base'], **cfg['snn-train'], **cfg['snn-test'])
    # args.nprocs = torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = data_loaders.get_data_loaders(path=args.dataset_path, data=args.data, resize=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    models = get_models(args)
    i= 0
    path = args.model_path
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    models.load_state_dict(state_dict, strict=False)
    models = multi_to_single_step(models,reset_mode=args.reset_mode)
    models.to(device)
    evaluator = Evaluator(models,T=args.T,add_time_dim=args.add_time_dim)
    acc, loss = evaluator.evaluation(test_loader)
    # loss
    print(acc)
    # print(loss)
    acc, loss = evaluator.aoi_evaluation(test_loader)
    from snncutoff.utils import save_pickle
    save_pickle(loss,name='aoi_timestep',path=os.path.dirname(path))
if __name__ == '__main__':
   main()
