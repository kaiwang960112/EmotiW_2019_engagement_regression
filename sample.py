import os, sys, shutil
import random as rd
from os import listdir
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data
import pdb
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss

def load_imgs(image_list_file):
    print("lmk_path:",image_list_file)
    with open(image_list_file) as file:
        segment_feat = list()
        all_datas = list()
        segment_label = list()
        classification_label = list()
        for index, line in enumerate(file):
            line = line.strip()
            video_name, label = line.split(' ',1)
            # pdb.set_trace()
            openface_feat_path = video_name
            with open(openface_feat_path) as feature_file:
                for index_1,feature_line in enumerate(feature_file):
                    feature_line = feature_line.strip()
                    arr = feature_line.split(' ')
                    #pdb.set_trace()
                    if index_1 >=1200:
                        pdb.set_trace()
                    if len(arr) != 14:
                        pdb.set_trace()
                    #pdb.set_trace()
                    if index_1 >=0:
                        segment_feat.append(arr)
                        segment_label.append(float(label))
                        if float(label) ==0.0:
                            classification_label.append(int(0))
                        if float(label) == 0.33:
                            classification_label.append(int(1))
                        if float(label) == 0.66:
                            classification_label.append(int(2))
                        if float(label) == 1.0:
                            classification_label.append(int(3))
                segment_label_numpy = np.array(segment_label)
                classification_label_numpy = np.array(classification_label)
                segment_feat_numpy = np.array(segment_feat)
                #pdb.set_trace()
                #segment_feat_numpy = segment_feat_numpy[:,4:152]
                segment_feat_numpy=segment_feat_numpy.astype(np.float)
                #pdb.set_trace()
                # segment_feat_numpy -= np.mean(segment_feat_numpy, axis=0) # axis=0，计算每一列的均值
                # segment_feat_numpy = normalize(segment_feat_numpy, axis=0, norm='max')
                all_datas.append((segment_feat_numpy, segment_label_numpy[0], classification_label_numpy[0]))
                segment_feat = list()
                segment_label = list()
                classification_label = list()
    print("len(all_datas): ",len(all_datas)) 
    return all_datas
		
class MsCelebDataset(data.Dataset):
	def __init__(self, image_list_file):
		self.all_datas = load_imgs(image_list_file)
		
	def __getitem__(self, index):
		feature, label, classification_label = self.all_datas[index]
		return feature, label, classification_label
		
	def __len__(self):
		return len(self.all_datas)
		