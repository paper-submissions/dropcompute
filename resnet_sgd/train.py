"""Imagenet training, enabled for Gaudi training."""

import argparse
import os
import random
import shutil
import time
import enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.cuda import amp

from logging_utils import tensorboard as tb
from logging_utils import log
import logging

try:
    import dali_loader
except ModuleNotFoundError:
    pass

try:
    import habana_utils
    import data
    import habana_frameworks.torch.core as htcore
    from habana_frameworks.torch.distributed import hccl
    from habana_frameworks.torch.hpex import hmp
    from habana_frameworks.torch.hpex import optimizers
except ModuleNotFoundError:
    pass


class Device(enum.Enum):
    HPU, CUDA, CPU = range(3)


FIRST_TIMING_EPOCH = 3


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='local batch size (default: 256) per device. The'
                         'global batch size is batch_size * world_size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='hccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
mixed_precision_group = parser.add_mutually_exclusive_group()
mixed_precision_group.add_argument('--autocast', dest='is_autocast',
                                   action='store_true',
                                   help='enable autocast mode on Gaudi')
mixed_precision_group.add_argument('--hmp', dest='is_hmp', action='store_true',
                                   help='enable hmp mode')

parser.add_argument('--device', default='hpu', help='device')
parser.add_argument('--data-path', default='/data/pytorch/imagenet/ILSVRC2012/',
                    help='dataset')
parser.add_argument('--output-dir', default='.', help='path where to save')
parser.add_argument(
    "--cache-dataset",
    dest="cache_dataset",
    help="Cache the datasets for quicker initialization. It also serializes the"
         " transforms",
    action="store_true",
)
parser.add_argument(
    '--deterministic',
    action="store_true",
    help='Whether or not to make data loading deterministic;This does not make'
         ' execution deterministic'
)
parser.add_argument(
    '--dl-worker-type',
    default='HABANA',
    type=lambda x: x.upper(),
    choices=["MP", "HABANA"],
    help='select multiprocessing or habana accelerated'
)
parser.add_argument(
    '--channels-last',
    default='False',
    type=lambda x: x.lower() == 'true',
    help='Whether input is in channels last format.'
         'Any value other than True(case insensitive) disables channels-last'
)
parser.add_argument(
    '--save-checkpoint',
    action="store_true",
    help='If provided (True), saves model/checkpoint.'
)
parser.add_argument(
    '--broadcast-buffers',
    action="store_true",
    help='Disables syncing buffers of the model at beginning of the forward'
         ' function. (default: False)'
)
parser.add_argument(
    '--sync-bn',
    dest='sync_bn',
    action='store_true',
    help='Use sync batch norm (default: False)'
)

# Drop compute arguments
parser.add_argument('--drop-rate', type=int, default=0,
                    help='frequency to drop batch between [0,100]. default 0.')

best_acc1 = 0


def init_distributed_mode(args):
    if args.device == Device.HPU:
        hccl.initialize_distributed_hpu()
    # To improve resnet dist performance, decrease number of all_reduce calls
    # to 1 by increasing bucket size to 230
    dist._DEFAULT_FIRST_BUCKET_BYTES = 230*1024*1024
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    logging.info('distributed initialization completed')


def zero_grad_hook(grad):
    return torch.zeros_like(grad)


def add_weight_decay(model, weight_decay):
    """Adds weights decay to all modules besides batch-norm."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if '.bn' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

    args.distributed = args.world_size > 1

    ngpus_per_node = torch.cuda.device_count() if args.device == 'gpu' else 8
    if args.rank <= 0:
        log.setup_logging(os.path.join(args.output_dir, 'Logger.INFO'))

    if torch.cuda.is_available():
        if args.rank <= 0:
            logging.info('CUDA detected. Setting device to GPU.')
        logging.info(torch.cuda.get_device_properties(0))
        args.device = Device.CUDA
        args.dist_backend = 'nccl'
    elif args.device == 'hpu':
        args.device = Device.HPU
        args.dist_backend = 'hccl'
    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    global best_acc1
    logging.info(args)

    device = torch.device(args.device.name.lower())

    if args.distributed:
        init_distributed_mode(args)
        habana_utils.setup_for_distributed(args.rank == 0)

    # create model
    if args.pretrained:
        logging.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logging.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](zero_init_residual=True)

    if args.distributed:
        if args.device == Device.CUDA:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                broadcast_buffers=args.broadcast_buffers)
        if args.device == Device.HPU:
            model.to(device)
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                broadcast_buffers=args.broadcast_buffers,
                gradient_as_bucket_view=True)
    # Data loading code
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    if args.device == Device.CUDA:
        train_loader = dali_loader.imagenet_loader(
            dataset=train_dir,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.workers,
            is_training=True,
            cpu=False,
        )

        val_loader = dali_loader.imagenet_loader(
            dataset=val_dir,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.workers,
            is_training=False,
            cpu=False
        )
    if args.device == Device.HPU:
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

        dataset, dataset_test, train_sampler, test_sampler = data.load_data(
            train_dir, val_dir, args)
        train_loader = data.get_data_loader(
            dataset,
            args.dl_worker_type,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            pin_memory_device='hpu'
        )

        val_loader = data.get_data_loader(
            dataset_test,
            args.dl_worker_type,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True,
            pin_memory_device='hpu'
        )

    criterion = nn.CrossEntropyLoss()
    if args.device == Device.CUDA:
        criterion = criterion.cuda(args.local_rank)

    sgd_optimizer = (
        optimizers.FusedSGD if args.device == Device.HPU else torch.optim.SGD)

    if args.weight_decay > 0:
        logging.info('filtering batch norm from weight decay parameters.')
        parameters = add_weight_decay(model, args.weight_decay)
    else:
        parameters = model.parameters()

    optimizer = sgd_optimizer(
        parameters,
        lr=args.lr,
        momentum=args.momentum,
    )

    # follow regime from Goyal et al. (ImageNet in 1 hour)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 80], gamma=0.1)
    lr_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1 / args.lr,
        total_iters=5 * len(train_loader))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1'].to(args.device)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    if args.rank <= 0 and not args.debug:
        tb.init(args.output_dir)
        tb.log_hyper_parameters(vars(args))

    train_time = 0
    rank_random = random.Random(dist.get_rank())

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.dl_worker_type == 'MP':
            train_sampler.set_epoch(epoch)

        if epoch == FIRST_TIMING_EPOCH:
            train_time = time.time()

        train(train_loader, model, criterion, optimizer, epoch, device,
              lr_warmup_scheduler, rank_random, args)
        acc1 = validate(val_loader, model, criterion, device, args)
        lr_scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.rank % ngpus_per_node == 0 and args.save_checkpoint:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, is_best)
        if dist.get_rank() <= 0 and not args.debug:
            tb.log_scalars(
                epoch,
                val_error_epoch=(100 - acc1),
                learning_rate=torch.tensor(lr_scheduler.get_last_lr()[0]))


def train(train_loader, model, criterion, optimizer, epoch, device,
          lr_warmup_scheduler, device_random, args):
    batch_time = AverageMeter('Time', device, ':6.3f')
    data_time = AverageMeter('Data', device, ':6.3f')
    losses = AverageMeter('Loss', device, ':.4e')
    top1 = AverageMeter('Acc@1', device, ':6.2f')
    top5 = AverageMeter('Acc@5', device, ':6.2f')
    drop_meter = AverageMeter('Drop', device, ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, drop_meter, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    dropped = False
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        if device_random.randint(1, 100) <= args.drop_rate and epoch > 1:
            logging.debug('rank %s dropped at epoch %s and step %s',
                          dist.get_rank(), epoch, i+1)
            hook_handles = []
            dropped = True
            for param in model.parameters():
                hook_handles.append(param.register_hook(zero_grad_hook))

        if args.device == Device.CUDA:
            with amp.autocast(dtype=torch.float16):
                output = model(images)
                loss = criterion(output, target)
        if args.device == Device.HPU:
            with torch.autocast(device_type='hpu', dtype=torch.bfloat16,
                                enabled=args.is_autocast):
                output = model(images)
                loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.device == Device.HPU:  # lazy mode
            htcore.mark_step()

        drop_count = torch.tensor([dropped], dtype=torch.float32).to(device)
        dist.all_reduce(drop_count)
        drop_meter.update(drop_count.item())

        optimizer.step()

        if args.device == Device.HPU:  # lazy mode
            htcore.mark_step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            progress.display(i + 1)
        lr_warmup_scheduler.step()

        if dropped:
            for handle in hook_handles:
                handle.remove()
            dropped = False

    if args.distributed:
        top1.all_reduce()
        losses.all_reduce()

    if dist.get_rank() <= 0 and not args.debug:
        tb.log_scalars(epoch,
                       train_error_epoch=(100 - top1.avg),
                       train_loss=losses.avg)


def validate(val_loader, model, criterion, device, args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                if args.channels_last:
                    image = image.contiguous(memory_format=torch.channels_last)

                # compute output
                if args.device == Device.CUDA:
                    with amp.autocast(dtype=torch.float16):
                        output = model(images)
                        loss = criterion(output, target)
                if args.device == Device.HPU:
                    with torch.autocast(device_type='hpu', dtype=torch.bfloat16,
                                        enabled=args.is_autocast):
                        output = model(images)
                        loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', device, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', device, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', device, ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', device, ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    progress.display_summary()
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class Summary(enum.Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, device, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.tensor([self.sum, self.count],
                             dtype=torch.float32).to(self.device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logging.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for each k in topk."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
