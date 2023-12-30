import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import hydra

from snncutoff.models.vggsnns import *
from snncutoff import data_loaders
import numpy as np
from snncutoff.configs import *
from omegaconf import DictConfig
from snncutoff.Evaluator import Evaluator
from snncutoff.utils import multi_to_single_step
from snncutoff import get_snn_model
from snncutoff.snncase import SNNCASE
from snncutoff.utils import save_pickle
import torch.backends.cudnn as cudnn
from snncutoff.utils import seed_all

@hydra.main(version_base=None, config_path='../configs', config_name='default')
def main(cfg: DictConfig):
    args = TestConfig(**cfg['base'], **cfg['snn-train'], **cfg['snn-test'])
    if args.seed is not None:
        seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = data_loaders.get_data_loaders(path=args.dataset_path, data=args.data, resize=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    models = get_snn_model(args)
    i= 0
    path = args.model_path
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    models.load_state_dict(state_dict, strict=False)
    models = multi_to_single_step(models,reset_mode=args.reset_mode)
    models.to(device)
    evaluator = Evaluator(models,args=args)
    acc, loss = evaluator.evaluation(test_loader)
    print(acc)
    print(np.mean(loss))
    acc, loss = evaluator.aoi_evaluation(test_loader)
    print(acc)
    print(np.mean(loss))
    save_pickle(loss,name='aoi_timestep',path=os.path.dirname(path))

    acc, loss, conf = evaluator.cutoff_evaluation(test_loader,train_loader=train_loader)
    # loss
    print(acc)
    print(np.mean(loss))
    # acc, loss = evaluator.aoi_evaluation(test_loader)
    save_pickle(loss,name='cutoff_timestep',path=os.path.dirname(path))
    save_pickle(conf,name='conf_timestep',path=os.path.dirname(path))
if __name__ == '__main__':
   main()
