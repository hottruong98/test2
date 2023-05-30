import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter

import video_transforms
import models
import datasets
from opt.AdamW import AdamW

model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))
# dataset_names = sorted(name for i in datasets.__all__)s

parser = argparse.ArgumentParser(description='3DCNN plus Attention Mechanism for Action Recognition')
parser.add_argument('--settings', default='./datasets/hmdb51/annotations_v2', metavar='DIR')
parser.add_argument('--dataset', default='hmdb51', choices=['hmdb51', 'hmdb51'])
parser.add_argument('--arch', default='rgb_r2plus1d_32f_34_bert10', choices=model_names)
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--iter-size', default=16, type=int)
parser.add_argument('--workers', default=2, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-3, type=float)
parser.add_argument('--print-freq', default=400, type=int)
parser.add_argument('--save-freq', default=1, type=int)
parser.add_argument('--num-seg', default=1, type=int)
parser.add_argument('--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--continue', dest='conti', action='store_true')

best_acc1 = 0 # best top 1 accuracy
best_loss = 30
warmup_epoch = 5

smt_pretrained = False

HALF = False
training_continue = False

def main():
    global args, best_acc, model, writer, best_loss, length, width, height, input_size, scheduler
    args = parser.parse_args()
    training_continue = args.conti
    scale = 0.5 # 'r2plus1d'
    input_size = int(224 * scale)
    width = int(340 * scale)
    height = int(256 * scale)

    save_location = './checkpoint/' + args.dataset + '_' + args.arch + '_split' + str(args.split)
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    writer = SummaryWriter(save_location)

    # ---> Create model
    if args.evaluate:
        print('Building validation model ...')
        model = build_model_validate()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif training_continue:
        model, start_epoch, optimizer, best_acc = build_model_continue()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print('Continuing with best accuracy: %.3f and start epoch %d and lr: %f' %(best_acc, start_epoch, lr))
    else:
        print('Building model with AdamW ...')
        model = build_model()
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_epoch = 0
    
    if HALF:
        model.half()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    print('Model %s is loaded. ' % (args.arch))

    # ---> Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.MSELoss().cuda()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    print('Saving everything to directory: %s.' % (save_location))
    if args.dataset == 'hmdb51':
        dataset = '/kaggle/input/hmdb51/HMDB51'
    elif args.dataset == 'ucf101':
        dataset = '/kaggle/input/hmdb51/HMDB51'
    else:
        print('No convenient dataset entered, exiting ...')
        return 0

    cudnn.benchmark = True
    modality=args.arch.split('_')[0]
    if '32f' in args.arch:
        length = 32
    elif '64f' in args.arch:
        length = 64
    else:
        length = 1
    modality = 'rgb'
    is_color = True
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    clip_mean = [0.43216, 107.7354, 99.4750] * args.num_seg * length
    clip_std = [0.22803, 0.22145, 0.216989] * args.num_seg * length

    normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)
    train_transform = video_transforms.Compose([
        video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor(),
        normalize,
    ])
    val_transform = video_transforms.Compose([
        video_transforms.CenterCrop((input_size)),
        video_transforms.ToTensor(),
        normalize,
    ])

    # ---> Data loading
    train_setting_file = 'train_%s_split%d.txt' % (modality, args.split)
    train_split_file = os.path.join(args.settings, train_setting_file)
    val_setting_file = 'val_%s_split%d.txt' % (modality, args.split)
    val_split_file = os.path.join(args.settings, val_setting_file)

    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print('No split file exists in %s directory. Prepare the dataset first' % (args.settings))


    train_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                    source=train_split_file,
                                                    phase='train',
                                                    modality=modality,
                                                    is_color=is_color,
                                                    new_length=length,
                                                    new_width=width,
                                                    new_height=height,
                                                    video_transform=train_transform,
                                                    num_segments=args.num_seg)
    val_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                 source=val_split_file,
                                                 phase='val',
                                                 modality=modality,
                                                 is_color=is_color,
                                                 new_length=length,
                                                 new_width=width,
                                                 new_height=height,
                                                 video_transform=val_transform,
                                                 num_segments=args.num_seg)
    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                           len(train_dataset),
                                                                           len(val_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)
    if args.evaluate:
        acc1, acc3, cls_loss = validate(val_dataset, model, criterion, criterion2, modality)
        return
    
    # ---> Start training model (from 0 or from previous trained model)
    for epoch in range(start_epoch, args.epochs):
        train(train_loader, model, criterion, criterion2, optimizer, epoch, modality)
        acc1 = 0.0
        cls_loss = 0
        if (epoch+1) % args.save_freq == 0:
            acc1, acc3, cls_loss = validate(val_loader, model, criterion, criterion2, modality)
            writer.add_scalar('data/top1_validation', acc1, epoch)
            writer.add_scalar('data/top3_validation', acc3, epoch)
            writer.add_scalar('data/classification_loss_validation', cls_loss, epoch)
            scheduler.step(cls_loss)
        
        is_best = acc1 >= best_acc1
        best_acc1 = max(acc1, best_acc1)

        if (epoch+1) % args.save_freq == 0:
            checkpoint_name = '%03d_%s' % (epoch+1, 'checkpoint.pth.tar')
            if is_best:
                print('Saving checkpoint ...')
                save_checkpoint({'epoch': epoch + 1,
                                 'arch': args.arch,
                                 'state_dict': model.state_dict(),
                                 'best_acc1': best_acc1,
                                 'best_loss': best_loss,
                                 'optimizer': optimizer.state_dict()},
                                 is_best,
                                 checkpoint_name,
                                 save_location)
                
    # ---> Save the last checkpoint
    checkpoint_name = '%03d_%s' % (epoch + 1, 'checkpoint.pth.tar')
    save_checkpoint({'epoch': epoch + 1,
                     'arch': args.arch,
                     'state_dict': model.state_dict(),
                     'best_acc1': best_acc1,
                     'best_loss': best_loss,
                     'optimizer': optimizer.state_dict()},
                     is_best,
                     checkpoint_name,
                     save_location)
    writer.export_scalars_to_json('./all_scalars.json')
    writer.close()

def build_model():
    modality = 'rgb'
    model_path = ''
    if args.dataset == 'ucf101':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](model_path=model_path, num_classes=101, length=args.num_seg)
    elif args.dataset == 'hmdb51':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](model_path=model_path, num_classes=51, length=args.num_seg)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.cuda()
    return model

def build_model_continue():
    model_location = './checkpoint/' + args.dataset + '_' + args.arch + '_split' + str(args.split)
    model_path = os.path.join(model_location, 'best_model.pth.tar')
    # Load all params save in previous trained model
    params = torch.load(model_path) 
    print(model_location)
    if args.dataset == 'ucf101':
        model = models.__dict__[args.arch](model_path='', num_classes=101, length=args.num_seg)
    elif args.dataset == 'hmdb51':
        model = models.__dict__[args.arch](model_path='', num_classes=51, length=args.num_seg)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(params['state_dict'])
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.load_state_dict(params['optimizer'])

    start_epoch = params['epoch']
    best_acc = params['best_acc1']
    return model, start_epoch, optimizer, best_acc

def build_model_validate():
    model_location = './checkpoint/' + args.dataset + '_' + args.arch + '_split' + str(args.split)
    model_path = os.path.join(model_location, 'best_model.pth.tar')
    params = torch.load(model_path)
    print(model_location)
    if args.dataset == 'ucf101':
        model = models.__dict__[args.arch](model_path='', num_classes=101, length=args.num_seg)
    elif args.dataset == 'hmdb51':
        model = models.__dict__[args.arch](model_path='', num_classes=51, length=args.num_seg)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval()
    return model

def train(train_loader, model, criterion, criterion2, optimizer, epoch, modality):
    batch_time = AverageMeter()
    cls_losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.train() # Switch to train mode
    end = time.time()
    optimizer.zero_grad()
    mini_batch_cls_loss = 0.0
    mini_batch_top1 = 0.0
    mini_batch_top3 = 0.0
    sample_per_iter = 0

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.view(-1, length, 3, input_size, input_size).transpose(1,2)
        if HALF:
            inputs = inputs.cuda().half()
        else:
            inputs = inputs.cuda()

        targets = targets.cuda()
        output, input_vectors, sequence_out, mask_sample = model(inputs)

        acc1, acc3 = accuracy(output.data, targets, topk=(1,3))
        mini_batch_top1 += acc1.item()
        mini_batch_top3 += acc3.item()

        cls_loss = criterion(output, targets)
        cls_loss = cls_loss / args.iter_size
        
        total_loss = cls_loss # total loss = lossMSE
        mini_batch_cls_loss += cls_loss.data.item() # total loss = lossMSE + cls_loss
        total_loss.backward()
        sample_per_iter += output.size(0)

        if (i+1) % args.iter_size == 0:
            # ---> Compute gradient & do SGD step
            optimizer.step()
            optimizer.zero_grad()
            cls_losses.update(mini_batch_cls_loss, sample_per_iter)
            top1.update(mini_batch_top1 / args.iter_size, sample_per_iter)
            top3.update(mini_batch_top3 / args.iter_size, sample_per_iter)
            batch_time.update(time.time() - end)
            end = time.time()
            mini_batch_cls_loss = 0
            mini_batch_top1 = 0.0
            mini_batch_top3 = 0.0
            sample_per_iter = 0
        
        if (i+1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f' %(i, batch_time.avg, cls_losses.avg))
    
    print('* Epoch: {epoch} Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Classification loss {cls_loss.avg:.4f}\n'
          .format(epoch=epoch, top1=top1, top3=top3, cls_loss=cls_losses))
    
    writer.add_scalar('data/cls_loss_training', cls_losses.avg, epoch)
    writer.add_scalar('data/top1_training', top1.avg, epoch)
    writer.add_scalar('data/top3_training', top3.avg, epoch)
def validate(val_loader, model, criterion, criterion2, modality):
    batch_time = AverageMeter()
    cls_losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.view(-1, length, 3, input_size).transpose(1, 2)
            if HALF:
                inputs = inputs.cuda().half()
            else:
                inputs = inputs.cuda()
            targets = targets.cuda()

            output, input_vectors, sequence_out, _ = model(inputs)
            cls_loss = criterion(output, targets)
            acc1, acc3 = accuracy(output.data, targets, topk=(1,3))
            cls_losses.update(cls_loss.data.item(), output.size(0))
            top1.update(acc1.item(), output.size(0))
            top3.update(acc3.item(), output.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
        print(' * * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Classification loss {cls_loss.avg:.4f}\n'
              .format(top1=top1, top3=top3, cls_loss=cls_losses))
    return top1.avg, top3.avg, cls_losses.avg

def save_checkpoint(state, is_best, file_name, resume_path):
    latest_path = os.path.join(resume_path, file_name)
    torch.save(state, latest_path)
    best_path = os.path.join(resume_path, 'best_model.pth.tar')
    if is_best:
        shutil.copyfile(latest_path, best_path)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print('Current learning rate is %4.6f:' % lr)
    for param_group  in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_v2(optimizer, epoch):
    is_warmup = epoch < warmup_epoch
    decay_rate = 0.2
    if is_warmup:
        lr = args.lr * (epoch+1) / warmup_epoch
    else:
        lr = args.lr * decay_rate ** (epoch+1-warmup_epoch)
    
    print('Current learning rate is %4.6f:' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_v3(optimizer, learning_rate_index):
    """Initial LR decayed by 10 every 150 epochs"""
    decay = 0.1 ** learning_rate_index
    lr = args.lr * decay
    print('Current learning rate is %4.8f:' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Compute the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
