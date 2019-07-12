import argparse
import os,sys,shutil
import time
# from sampler import ImbalancedDatasetSampler
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import math
from lstm_module import lstm_processing
from sample import MsCelebDataset
import scipy.io as sio  
import numpy as np
import pdb
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='/media/sdb/xxzeng/MS-1m-subset-cleanning-by-demeng-size-178_218/MS-1m-subset-cleanning-by-demeng-size-178_218/', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='our_lstm_attention', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=12, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-b_t', '--batch-size_t', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='../../Data/Model/resnet34_ferplus.pkl', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model_dir','-m', default='./engagement_regression', type=str)
parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')

best_prec1 = 1


def main():
    global args, best_prec1
    args = parser.parse_args()
    print('end2end?:', args.end2end)
    train_list_file = 'openpose_jianfei_split_train.txt'
    train_dataset =  MsCelebDataset(train_list_file)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)
    val_list_file = 'openpose_jianfei_split_validation.txt'
    val_dataset =  MsCelebDataset(val_list_file)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        # sampler=ImbalancedDatasetSampler(val_dataset),
        batch_size=args.batch_size_t,
        num_workers=args.workers, pin_memory=True)

    #assert(train_dataset.max_label == val_dataset.max_label)

    
    # prepare model
    model = None
    assert(args.arch in ['our_lstm_attention','resnet18','resnet34','resnet101'])
    if args.arch == 'our_lstm_attention':
        #model = Res()
        model = lstm_processing(feature_num=14, hidden_dim=512)
        print("we use our lstm attention network!")
   #     model = resnet18(pretrained=False, nverts=nverts_var,faces=faces_var,shapeMU=shapeMU_var,shapePC=shapePC_var,num_classes=class_num, end2end=args.end2end)
    if args.arch == 'resnet34':
        model = resnet34(end2end=args.end2end)
    if args.arch == 'resnet101':
        pass
    #    model = resnet101(pretrained=False,nverts=nverts_var,faces=faces_var,shapeMU=shapeMU_var,shapePC=shapePC_var, num_classes=class_num, end2end=args.end2end)
    
#    for param in model.parameters():
#        param.requires_grad = False

    
    # for name, p in model.named_parameters():
    #      if not ( 'fc' in name or 'alpha' in name or 'beta' in name):
             
    #          p.requires_grad = False
    #      else:
    #          print 'updating layer :',name
    #          print p.requires_grad
           
#    for param_flow in model.module.resnet18_optical_flow.parameters():
#        param_flow.requires_grad =True
    model = torch.nn.DataParallel(model).cuda()
    #pdb.set_trace()
    #model.module.theta.requires_grad = True
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion1 = nn.MSELoss().cuda()
    criterion2 = kw_rank_loss().cuda()
    #criterion=Cross_Entropy_Sample_Weight.CrossEntropyLoss_weight().cuda()
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

   # optionally resume from a checkpoint
    cudnn.benchmark = True
    print ('args.evaluate',args.evaluate)
    if args.evaluate:
        validate(val_loader, model, criterion1, criterion, criterion2)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion1, criterion,optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion1, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        
        best_prec1 = min(prec1.cuda()[0], best_prec1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best[0])

def train(train_loader, model, criterion1, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    classification_losses = AverageMeter()
    regression_losses = AverageMeter()
    rank_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (feature, label,classification_label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        input = feature.float()
        
        target = label.float()
        target = target.cuda(async=True)
        target_class = classification_label
        target_class = target_class.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).unsqueeze(1)
        target_class_var = torch.autograd.Variable(target_class)
        
        
        # compute output
        #pdb.set_trace()
        class_score, pred_score = model(input_var)
        #pdb.set_trace()
        regression_loss = criterion1(pred_score, target_var)
        classification_loss = criterion(class_score, target_class_var)
        # pdb.set_trace()
        #rank_loss = criterion2(feature_results, target_var)
        loss = regression_loss + 0*classification_loss
        prec1 = accuracy(class_score.data, target_class_var, topk=(1,))
        # pdb.set_trace()
        regression_losses.update(regression_loss.item(), input.size(0))
        classification_losses.update(classification_loss.item(), input.size(0))
        #rank_losses.update(rank_loss, input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'Data {data_time.val} ({data_time.avg})\t'
                  'regLoss {regression_loss.val} ({regression_loss.avg})\t'
                  'claLoss {classification_loss.val} ({classification_loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'
                                                              .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, regression_loss=regression_losses, classification_loss=classification_losses,top1=top1))


def validate(val_loader, model, criterion1, criterion):
    batch_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    classification_losses = AverageMeter()
    regression_losses = AverageMeter()
    rank_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (feature, label, classification_label) in enumerate(val_loader):
        # target = target.cuda(async=True)
        # input_var = torch.autograd.Variable(input, volatile=True)
        # target_var = torch.autograd.Variable(target, volatile=True)
        # compute output
        input = feature.float()
        
        target = label.float()
        target = target.cuda(async=True)
        target_class = classification_label
        target_class = target_class.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).unsqueeze(1)
        target_class_var = torch.autograd.Variable(target_class)
        #gr_em_label = gr_em_label.cuda(async=True)
        #gr_em_label_var = torch.autograd.Variable(gr_em_label)

        class_score, pred_score = model(input_var)
        #pdb.set_trace()
        regression_loss = criterion1(pred_score, target_var)
        classification_loss = criterion(class_score, target_class_var)
        # pdb.set_trace()
        #rank_loss = criterion2(feature_results, target_var)
        loss = regression_loss + 0*classification_loss
        # loss = regression_loss
        # loss = classification_loss*(1/math.exp(-regression_loss))

        # measure accuracy and record loss
        prec1 = accuracy(class_score.data, target_class_var, topk=(1,))
        regression_losses.update(regression_loss.data[0], input.size(0))
        classification_losses.update(classification_loss.item(), input.size(0))
        #rank_losses.update(rank_loss, input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val} ({batch_time.avg})\t'
                  'regLoss {regression_loss.val} ({regression_loss.avg})\t'
                  'claLoss {classification_loss.val} ({classification_loss.avg})\t'
                  'Prec@1 {top1.val} ({top1.avg})\t'
                  .format(
                   i, len(val_loader), batch_time=batch_time, regression_loss=regression_losses, classification_loss = classification_losses, top1=top1))

    print(' * Prec@1 {regression_loss.avg} '
          .format(regression_loss=regression_losses))

    return regression_losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    full_filename = os.path.join(args.model_dir, filename)
    full_bestname = os.path.join(args.model_dir, 'model_best.pth.tar')
    torch.save(state, full_filename)
    epoch_num = state['epoch']
    if epoch_num%1==0 and epoch_num>=0:
        torch.save(state, full_filename.replace('checkpoint','checkpoint_'+str(epoch_num)))
    if is_best:
        shutil.copyfile(full_filename, full_bestname)


class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [int(args.epochs*0.3), int(args.epochs*0.5), int(args.epochs*0.8)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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




class kw_rank_loss(nn.Module):    
    def __init__(self):        
        super(kw_rank_loss, self).__init__()          
    def forward(self, feature_results, target_var):
        engagement_level_1 = np.zeros((1, feature_results.shape[1]))
        engagement_level_2 = np.zeros((1, feature_results.shape[1]))
        engagement_level_3 = np.zeros((1, feature_results.shape[1]))
        engagement_level_4 = np.zeros((1, feature_results.shape[1]))
        level_1_num = level_2_num = level_3_num = level_4_num = 1
        #pdb.set_trace()
        size = feature_results.shape[0]
        feature_results = feature_results.cpu().data.numpy()
        margin = 0.75
        #pdb.set_trace()
        for i in range(size):
            if target_var[i] <= 0.1:
                engagement_level_1 += feature_results[i]
                level_1_num += 1
            if 0.3 <target_var[i] <= 0.4:
                engagement_level_2 += feature_results[i]
                level_2_num += 1
            if 0.6 <target_var[i] <= 0.7:
                engagement_level_3 += feature_results[i]
                level_3_num += 1
            if 0.8 <target_var[i] <= 1:
                engagement_level_4 += feature_results[i]
                level_4_num += 1
        #pdb.set_trace()
        engagement_level_1 = engagement_level_1/level_1_num
        engagement_level_2 = engagement_level_2/level_2_num
        engagement_level_3 = engagement_level_3/level_3_num
        engagement_level_4 = engagement_level_4/level_4_num
        #pdb.set_trace()
        dist_1_2 = np.linalg.norm(engagement_level_1 - engagement_level_2)
        dist_1_3 = np.linalg.norm(engagement_level_1 - engagement_level_3)
        dist_1_4 = np.linalg.norm(engagement_level_1 - engagement_level_4)
        dist_2_3 = np.linalg.norm(engagement_level_2 - engagement_level_3)
        dist_2_4 = np.linalg.norm(engagement_level_2 - engagement_level_4)
        dist_3_4 = np.linalg.norm(engagement_level_3 - engagement_level_4)
        #pdb.set_trace()
        loss = max(0.0,(dist_1_2 - dist_1_3 + margin)) + max(0.0,(dist_1_2 - dist_1_4 +2*margin)) + \
               max(0.0,(dist_2_3 - dist_2_4 + margin)) + max(0.0,(dist_2_3 - dist_1_4 +2*margin)) + \
               max(0.0,(dist_3_4 - dist_2_4 + margin)) + max(0.0,(dist_3_4 - dist_1_4 +2*margin))       
        return loss

if __name__ == '__main__':
    main()
