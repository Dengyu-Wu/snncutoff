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
from snncutoff.utils import multi_to_single_step, preprocess_ann_arch
from snncutoff.API import get_model
from snncutoff.utils import save_pickle
import torch.backends.cudnn as cudnn
from snncutoff.utils import set_seed

@hydra.main(version_base=None, config_path='../configs', config_name='test')
def main(cfg: DictConfig):
    args = TestConfig(**cfg['base'], **cfg['snn-train'], **cfg['snn-test'])
    if args.seed is not None:
        set_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = data_loaders.get_data_loaders(path=args.dataset_path, data=args.data, transform=False,resize=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    models = get_model(args)
    i= 0
    path = args.model_path
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    models.load_state_dict(state_dict, strict=False)
    if not args.multistep:
        if not args.multistep_ann:
            models = preprocess_ann_arch(models)
        models = multi_to_single_step(models, args.multistep_ann, reset_mode=args.reset_mode)
    models.to(device)
    evaluator = Evaluator(models,args=args)
    acc, loss = evaluator.evaluation(test_loader)
    print(acc)
    print(np.mean(loss))
    result={'accuracy': acc}
    save_pickle(result,name='timestep_cutoff',path=os.path.dirname(path))

    acc, timesteps = evaluator.oct_evaluation(test_loader)
    print(acc)
    print(np.mean(timesteps))
    result={'accuracy': acc, 'timesteps': timesteps}
    save_pickle(result,name='oct_timestep',path=os.path.dirname(path))

    acc=[]
    timesteps=[]
    samples_number=[]
    for i in range(1):
        epsilon = 0.05*i
        _acc, _timesteps, _samples_number = evaluator.cutoff_evaluation(test_loader,train_loader=train_loader,epsilon=epsilon)
        acc.append(_acc)
        timesteps.append(_timesteps)
        samples_number.append(_samples_number)
    acc = np.array(acc)
    timesteps = np.array(timesteps)
    result={'accuracy': acc, 'timesteps': timesteps, 'samples_number':samples_number}
    save_pickle(result,name='topk_cutoff',path=os.path.dirname(path))

if __name__ == '__main__':
   main()
