import os
import torch
import warnings
import torch.optim
import hydra

from snncutoff import data_loaders
import numpy as np
from snncutoff.configs import *
from omegaconf import DictConfig
from snncutoff.Evaluator import Evaluator
from snncutoff.utils import multi_to_single_step
from snncutoff import get_snn_model
from snncutoff.utils import save_pickle
import torch.backends.cudnn as cudnn
from snncutoff.utils import seed_all

@hydra.main(version_base=None, config_path='../configs', config_name='test')
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
    if not args.multistep:
        models = multi_to_single_step(models,reset_mode=args.reset_mode)
    models.to(device)
    evaluator = Evaluator(models,args=args)
    acc, loss = evaluator.evaluation(test_loader)
    print(acc)
    print(np.mean(loss))
    save_pickle(acc,name='timestep_cutoff',path=os.path.dirname(path))

    acc, timesteps = evaluator.aoi_evaluation(test_loader)
    print(acc)
    print(np.mean(timesteps))
    save_pickle(timesteps,name='aoi_timestep',path=os.path.dirname(path))

    acc, timesteps, conf = evaluator.cutoff_evaluation(test_loader,train_loader=train_loader)
    # loss
    print(acc)
    print(np.mean(timesteps))
    # acc, loss = evaluator.aoi_evaluation(test_loader)
    save_pickle(timesteps,name='topk_cutoff_sampels',path=os.path.dirname(path))
    save_pickle(conf,name='conf_timestep',path=os.path.dirname(path))

    acc=[]
    for i in range(10):
        evaluator.args.sigma = 1-0.01*i
        _acc, timesteps, conf = evaluator.cutoff_evaluation(test_loader,train_loader=train_loader)
        acc.append(_acc)
    acc = np.array(acc)
    save_pickle(acc,name='topk_cutoff',path=os.path.dirname(path))

if __name__ == '__main__':
   main()
