import argparse
import os,sys,shutil
import time

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
from sample_test import MsCelebDataset
import scipy.io as sio  
import numpy as np
import pdb
from torch.autograd import Variable
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir_val', metavar='DIR', default='/media/sdc/kwang/ferplus/different_pose_ferplus/', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='our_lstm_attention', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./data/resnet18/checkpoint_3.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_dir','-m', default='./model', type=str)
parser.add_argument('--end2end', default=True,\
        help='if true, using end2end with dream block, else, using naive architecture')





def get_val_data():

    val_list_file = '/media/sdc/kwang/Emotiw_2019_task2/OpenPose_features2/train_openpose/openpose_jianfei_split_validation.txt'
    val_dataset =  MsCelebDataset(val_list_file)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        # sampler=ImbalancedDatasetSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)

    return val_loader


def main(arch,resume):
    global args
    args = parser.parse_args()
    arch = arch.split('_')[0]
    model = None
    model = None
    assert(args.arch in ['our_lstm_attention'])
    if args.arch == 'our_lstm_attention':
        model = lstm_processing(feature_num=14, hidden_dim=512)

    
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    
    assert(os.path.isfile(resume))
    checkpoint = torch.load(resume)
    
    model.load_state_dict(checkpoint['state_dict'], strict = False)

    cudnn.benchmark = True

    
    
    
    feat_dim = 256
    val_loader = get_val_data()
    video_file_names = []
    #video_file_names = torch.tensor(video_file_names).cuda()
    fp = open("task2_openpose_score.txt","w+")
    correct =0
    mse_value = 0
    video_num = 0
    for i, (feature,label,class_label, video_name) in enumerate(val_loader):
        #input = torch.tensor(input)
        print ('i',i)
        print ('video_name',video_name)
        # print label
        input = feature.float()
        label = label.float()
        label = label.cuda(async=True)
        video_num += 1	
        input_var = torch.autograd.Variable(input)
        # pdb.set_trace()
        class_score, output = model(input_var)
        # output_data = output.cpu().data.numpy()
        mse_value += (output.cpu().data.numpy()[0] - label[0])*(output.cpu().data.numpy()[0] - label[0])
        fp.write(str(label) +'/' + str(video_name) + ' ' + str(output.cpu().data.numpy()[0]) + '\n')
        print("prediction:", str(output.cpu().data.numpy()[0]))
        print("ground_truth:", str(label))
    print('final_mse :', mse_value/video_num)

if __name__ == '__main__':
    
    infos = [ ('resnet18_naive', '/media/sdc/kwang/Emotiw_2019_task2/OpenPose_features2/train_openpose/engagement_regression/model_best.pth.tar'), 
               ]


    for arch, model_path in infos:
        print("{} {}".format(arch, model_path))
        main(arch, model_path)
        
        print()