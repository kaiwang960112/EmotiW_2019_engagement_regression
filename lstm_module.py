import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pdb
class lstm_processing(nn.Module):
    def __init__(self, feature_num=14, hidden_dim=512): #hidden_dim=1024
        ''' define the LSTM regression network '''
        super(lstm_processing, self).__init__()
        # self.lstm = nn.LSTM(feature_num, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(feature_num, hidden_dim, num_layers=2, batch_first=True)
        # self.lstm = nn.LSTM(feature_num, hidden_dim, 1, batch_first=True)
        self.dense = torch.nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(1024, 512),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
        )
        self.alpha = nn.Sequential(nn.Linear(128, 1),
                                   nn.Sigmoid())
        '''self.classify = torch.nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())'''
        self.regression = torch.nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid())
        self.regression_1 = torch.nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid())
        self.classify = torch.nn.Sequential(
            nn.Linear(128, 4))
    def forward(self, inputs):
        #pdb.set_trace()
        #inputs = inputs.unsqueeze(1)

        #print("inputs:: ",inputs.shape)
        #pdb.set_trace()
        ft_s = []
        alphas = []
        for i in range(10):
            #pdb.set_trace()
            #print("input-shape: ", inputs[:, i*75 : (i+1)*75,:].shape)
            #pdb.set_trace()


            output,_ = self.lstm(inputs[:, i*120 : (i+1)*120,:])
            #print("output: ",output.shape)
            #pdb.set_trace()
            ft = self.dense(output)[:, 119, :]

            ft_s.append(ft)
            alphas.append(self.alpha(ft))

        ft_s_stack = torch.stack(ft_s, dim=2)
        #pdb.set_trace()
        alphas_stack = torch.stack(alphas, dim=2)
        ft_final = ft_s_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))

        output = self.classify(ft_final)
        regression_value = self.regression(output)
        regression_value_1 = self.regression_1(ft_final)
        #pdb.set_trace()
        #output = self.dense(output[:,-1,:])
        return output, (regression_value+regression_value_1)/2