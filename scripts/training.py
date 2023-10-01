import argparse
import shutil
import os
import time
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from easycutoff.models.resnet_models import resnet19
from easycutoff.models.VGG_models import *
from easycutoff import dvs_loaders
from easycutoff.functions import TET_loss
from easycutoff.utils import seed_all, get_logger, OutputHook, sethook
from easycutoff.ddp import reduce_mean, ProgressMeter, adjust_learning_rate, accuracy, AverageMeter

from configs import BaseConfig, SNNConfig, AllConfig
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

def main_worker(local_rank, args):
    args.local_rank = local_rank
    if args.seed is not None:
        seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc1 = .0

    dist.init_process_group(backend='nccl',
                            init_method="tcp://localhost:"+args.port,
                            world_size=args.nprocs,
                            rank=args.local_rank)
    load_names = None
    save_names = None

    # load_names = 'dvs-cifar1048_V1_D05_VGGSNN_distribute_ensemble_gama05.pth'
    save_names = args.log+'/'+args.project + '.pth'
    # save_names = './saved_models/Cifar10DVS/'+args.log+'.pth'
    # model = VGGSNN_Spike() #if args.spike_out else VGGSNN()
    model = VGGSNN(current_out=True)
    #model = sew_resnet.__dict__['sew_resnet34'](T=args.T, connect_f='ADD')
    model.T = args.T

    if load_names != None:
        state_dict = torch.load(load_names)
        model.load_state_dict(state_dict, strict=False)

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
                                 weight_decay=5e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    cudnn.benchmark = True

    # Data loading code
    train_dataset, val_dataset = dvs_loaders.get_dvs_loaders(path='/LOCAL/dengyu/dvs_dataset/dvs-cifar10', resize=False)
    # train_dataset, val_dataset = data_loaders.build_dvscifar()
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

    # logger = get_logger('./saved_models/Cifar10DVS/'+args.log+'.log')
    logger = get_logger(args.log+'/'+ args.project + '.log')
    logger.info('start training!')

    if args.local_rank == 0:
        if args.wandb_logging:
            wandb.init(config=args,name=args.log, project=args.project, sync_tensorboard=True)
        if args.tensorboard_logging:
            writer = SummaryWriter(args.log)

    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        cs1, cs2 = train(train_loader, model, criterion, optimizer, epoch, local_rank,
              args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, local_rank, args)

        scheduler.step()
        # remember best acc@1 and save checkpoint
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)
        logger.info('Epoch:[{}/{}]\t best acc={:.5f}\t acc={:.3f} \t cs1={:.3f} \t cs2={:.3f}'.format(epoch+1 , args.epochs, best_acc1, acc1,cs1, cs2))
        if args.local_rank == 0:
            if args.wandb_logging:
                wandb.log({"accuracy": acc1, "cs1": cs1})
            if args.tensorboard_logging:
                writer.add_scalar("training/accuracy", acc1, epoch+1)
                writer.add_scalar("training/cs1", cs1, epoch+1)
        t2 = time.time()
        print('Time elapsed: ', t2 - t1)
        print('Best top-1 Acc: ', best_acc1)
        if is_best:
            if args.local_rank == 0:
                torch.save(model.module.state_dict(), save_names)
                #torch.save(model, './saved_models/Cifar10DVS/'+args.log+'.pt')
        # save_checkpoint(
        #     {
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.module.state_dict(),
        #         'best_acc1': best_acc1,
        #     }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    
    metric_list = ['Loss', 'Acc@1','Acc@5','cs1','cs2']
    metrics = []
    for metric in metric_list:
        metrics.append(AverageMeter(metric, ':.4e'))


    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # cs1 = AverageMeter('cs1@5', ':6.3f')
    # cs2 = AverageMeter('cs2@5', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                            #  [batch_time, data_time, losses, top1, top5],
                             [batch_time, data_time] + metrics,
                             prefix="Epoch: [{}]".format(epoch+1))
    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        inputs = images.transpose(0,1)
        output_hook = OutputHook()
        model = sethook(output_hook)(model)

        # compute output
        outputs = model(inputs)
        mean_out = outputs.mean(0)

        target_onehot  = torch.nn.functional.one_hot(target, num_classes=outputs.size()[-1]) 
        target_onehot  = target_onehot.to(torch.float32)
        
        
        _target = torch.unsqueeze(target,dim=0)  # T N C 
        right_predict_mask = outputs.max(-1)[1].eq(_target).to(torch.float32)
        tan_phi_mean = torch.stack(output_hook,dim=2).flatten(0, 1).contiguous() # T*N L C
        right_predict_mask = torch.unsqueeze(right_predict_mask,dim=2).flatten(0, 1).contiguous().detach()
        tan_phi_mean_masked = tan_phi_mean*right_predict_mask

        tan_phi_max = tan_phi_mean_masked.max(dim=0)[0] # find max
        tan_phi_min = (tan_phi_mean_masked+(1-right_predict_mask)*1000.0).min(dim=0)[0] # find min, exclude zero value
        tan_phi_min = tan_phi_min*tan_phi_min.lt(torch.tensor(1000.0)).to(torch.float32) # set wrong prediction to zero
        
        cs_loss = (tan_phi_max.detach()-tan_phi_min).pow(2).mean() #change pow into abs

        cs1_mean = tan_phi_min.mean()
        cs2_mean = tan_phi_max.mean()
        if not args.TET:
            loss = criterion(mean_out, target_onehot)
        else:
            loss = TET_loss(outputs, target_onehot, criterion, args.means, args.lamb)

        loss = loss  + args.alpha*(cs_loss)

        acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_list = [loss,acc1,acc5,cs1_mean,cs2_mean]
        reduced_metrics = []
        for reduced_metric in reduced_list:
            reduced_metrics.append(reduce_mean(reduced_metric, args.nprocs))

        # reduced_loss = reduce_mean(loss, args.nprocs)
        # reduced_acc1 = reduce_mean(acc1, args.nprocs)
        # reduced_acc5 = reduce_mean(acc5, args.nprocs)
        # reduced_cs1 = reduce_mean(cs1_mean, args.nprocs)
        # reduced_cs2 = reduce_mean(cs2_mean, args.nprocs)
        
        for metric,reduced_metric in zip(metrics, reduced_metrics):
            metric.update(reduced_metric.item(), images.size(0))
        # losses.update(reduced_loss.item(), images.size(0))
        # top1.update(reduced_acc1.item(), images.size(0))
        # top5.update(reduced_acc5.item(), images.size(0))
        # cs1.update(reduced_cs1.item(), images.size(0))
        # cs2.update(reduced_cs2.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        model = sethook(output_hook)(model,remove=True)

        if i % args.print_freq == 0:
            progress.display(i)

    return metrics[3].avg, metrics[4].avg

def validate(val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
    #                          prefix='Test: ')

    
    metric_list = ['Loss', 'Acc@1','Acc@5']
    metrics = []
    for metric in metric_list:
        metrics.append(AverageMeter(metric, ':.4e'))
    progress = ProgressMeter(len(val_loader),
                            #  [batch_time, losses, top1, top5],
                             [batch_time] + metrics,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            output = model(images.transpose(0,1))
            # output = output*torch.sigmoid(sigma)
            mean_out = torch.mean(output, dim=0)
            loss = criterion(mean_out, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

            torch.distributed.barrier()


            reduced_list = [loss,acc1,acc5]
            reduced_metrics = []
            for reduced_metric in reduced_list:
                reduced_metrics.append(reduce_mean(reduced_metric, args.nprocs))
            
            for metric,reduced_metric in zip(metrics, reduced_metrics):
                metric.update(reduced_metric.item(), images.size(0))

            # reduced_loss = reduce_mean(loss, args.nprocs)
            # reduced_acc1 = reduce_mean(acc1, args.nprocs)
            # reduced_acc5 = reduce_mean(acc5, args.nprocs)

            # losses.update(reduced_loss.item(), images.size(0))
            # top1.update(reduced_acc1.item(), images.size(0))
            # top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed timecs1.avg, cs2.avg
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
        #                                                             top5=top5))

    # return top1.avg
    return metrics[1].avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


@hydra.main(version_base=None, config_path='../configs', config_name='default')
def main(cfg: DictConfig):   
    all_conf = AllConfig(**cfg['base'],**cfg['snn-train'],**cfg['logging'])
    os.environ['CUDA_VISIBLE_DEVICES'] = all_conf.gpu_id
    all_conf.nprocs = torch.cuda.device_count()
    all_conf.log = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if all_conf.wandb_logging:
        wandb.login()


    mp.spawn(main_worker, nprocs=all_conf.nprocs, args=(all_conf,))


if __name__ == '__main__':
    main()