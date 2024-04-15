import argparse
import shutil
import os
import sys
import time
import warnings

import wandb
import hydra

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter
from snncutoff import data_loaders
from snncutoff.utils import set_seed, get_logger
from snncutoff.ddp import reduce_mean, ProgressMeter, accuracy, AverageMeter

from snncutoff.configs import AllConfig
from omegaconf import DictConfig
from snncutoff import SNNCASE
from snncutoff.API import get_model

def main_worker(local_rank, args):
    args.local_rank = local_rank
    if args.seed is not None:
        set_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc = .0

    dist.init_process_group(backend='nccl',
                            init_method="tcp://localhost:"+args.port,
                            world_size=args.nprocs,
                            rank=args.local_rank)

    save_names = args.log+'/'+args.project + '.pth'
    checkpoint_names = args.log+'/'+args.project + '_checkpoint.pth'
    bestpoint_names = args.log+'/'+args.project + '_bestpoint.pth'

    model = get_model(args)
    if args.checkpoint_path != 'none':
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.SGD(model.parameters(),
                                 args.lr,
                                 momentum=0.9,
                                 weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    if args.checkpoint_path != 'none':
        checkpoint = torch.load(args.checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    cudnn.benchmark = True

    # Data loading code
    train_dataset, val_dataset = data_loaders.get_data_loaders(path=args.dataset_path, data=args.data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=False,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler)
    


    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args)
        return

    logger = get_logger(args.log+'/'+ args.project + '.log')
    logger.info('start training!')

    if args.local_rank == 0:
        if args.wandb_logging:
            wandb.init(config=args,name=args.log, project=args.project)
        if args.tensorboard_logging:
            writer = SummaryWriter(args.log)


    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        # Create metric list
        training_metric_dic = {'Loss': [], 'Acc@1': [], 'Acc@5': []}
        custome_metric_dic  = {'cs_loss': []}
        training_metric_dic.update(custome_metric_dic)
        
        # train for one epoch
        training_metrics = train(train_loader, model, criterion, training_metric_dic, optimizer, epoch, local_rank, args)
        # evaluate on validation set
        test_metric_dic = {'Loss': [], 'Acc@1': [], 'Acc@5': []}
        custome_metric_dic  = {'cs_loss': []}
        test_metric_dic.update(custome_metric_dic)

        test_metrics = validate(val_loader, model, criterion, test_metric_dic, local_rank, args)
        scheduler.step()

        for name,training_metric in zip(list(training_metric_dic.keys()),training_metrics):
            training_metric_dic[name] = training_metric.avg

        for name,test_metric in zip(list(test_metric_dic.keys()),test_metrics):
            test_metric_dic[name] = test_metric.avg

        test_metric_dic['cs_loss'] = training_metric_dic['cs_loss']
        acc = test_metric_dic['Acc@1']
        is_best = acc >= best_acc
        best_acc = max(acc, best_acc)
        test_metric_dic['lr'] = scheduler.get_lr()[0]
        info_str = ', '.join(f"{key}: {value:.3f}" for key, value in test_metric_dic.items())
        logger.info('Epoch:[{}/{}]\t Best Acc={:.3f}\t'.format(epoch+1 , args.epochs, best_acc)+f"{info_str}")

        log_dic = {k:test_metric_dic[k] for k in ('Loss','Acc@1','cs_loss') if k in test_metric_dic}
        if args.local_rank == 0:
            if args.wandb_logging:
                wandb.log(log_dic)
            if args.tensorboard_logging:
                for key, value in log_dic.items():
                    writer.add_scalar('training/'+key, value, global_step=epoch)
        t2 = time.time()

        if is_best:
            if args.local_rank == 0:
                torch.save(model.module.state_dict(), save_names)

        if args.checkpoint_save:
            if args.local_rank == 0:
                save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_acc': best_acc,
                    }, is_best,checkpoint_names=checkpoint_names,bestpoint_names=bestpoint_names)


def train(train_loader, model, criterion, base_metrics, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    
    metrics = []
    # for metric in metric_list:
    for metric in list(base_metrics.keys()):
        metrics.append(AverageMeter(metric, ':.4e'))

    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time] + metrics,
                             prefix="Epoch: [{}]".format(epoch+1))
    # switch to train mode
    model.train()
    end = time.time()
    snncase = SNNCASE(net=model, criterion=criterion, args=args)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        
        regularization = args.regularizer != 'none'
        mean_out, loss = snncase.forward(images, target, regularization=regularization) 
        cs_loss = snncase.get_loss_reg()
        
        acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_list = [loss,acc1,acc5]
        #custom value
        if regularization:
            reduced_list.append(cs_loss)
        
        reduced_metrics = []
        for reduced_metric in reduced_list:
            reduced_metrics.append(reduce_mean(reduced_metric, args.nprocs))

        for metric,reduced_metric in zip(metrics, reduced_metrics):
            metric.update(reduced_metric.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if regularization:
            snncase.remove_hook()
            

        if i % args.print_freq == 0:
            progress.display(i)

    return metrics

def validate(val_loader, model, criterion, base_metrics, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')   
    metrics = []
    for metric in list(base_metrics.keys()):
        metrics.append(AverageMeter(metric, ':.4e'))
    progress = ProgressMeter(len(val_loader),
                             [batch_time] + metrics,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    

    with torch.no_grad():
        end = time.time()
        snncase = SNNCASE(net=model, criterion=criterion, args=args)
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)
            # compute output
 
            mean_out,loss = snncase.forward(images, target,regularization=False)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_list = [loss,acc1,acc5]
            reduced_metrics = []
            for reduced_metric in reduced_list:
                reduced_metrics.append(reduce_mean(reduced_metric, args.nprocs))
            
            for metric,reduced_metric in zip(metrics, reduced_metrics):
                metric.update(reduced_metric.item(), images.size(0))

            # measure elapsed timecs1.avg, cs2.avg
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return metrics


def save_checkpoint(state, is_best, checkpoint_names='checkpoint.pth',bestpoint_names='best_model.pth'):
    torch.save(state, checkpoint_names)
    if is_best:
        shutil.copyfile(checkpoint_names, bestpoint_names)


@hydra.main(version_base=None, config_path='../snncutoff/configs', config_name='default')
def main(cfg: DictConfig):   
    args = AllConfig(**cfg['base'],**cfg['snn-train'],**cfg['logging'])
    if args.gpu_id != 'none':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.nprocs = torch.cuda.device_count()
    args.log = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    #Model Name
    args.project = args.data
    
    if args.wandb_logging:
        wandb.login()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args,))


if __name__ == '__main__':
    main()