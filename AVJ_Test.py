#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import glob
import copy
from tqdm import *
import time
import numpy as np
from sklearn import preprocessing
# Choose Pytorch library for CNN
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import  Dataset, DataLoader

from pandas import DataFrame
from pandas.io.parsers import read_csv
import scipy.ndimage as ndi

import skimage.io as skio
import skimage.transform as skt

from torch.nn import functional as F
import torchvision.transforms.functional as TF
import random



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_HEIGHT = 128
INPUT_WIDTH = 128
OUTPUTS = 4
BATCH_SIZE = 1


def load_data(data_dir, data_csv, load_pts=True):
    
    df = read_csv(data_csv)  # load pandas dataframe
    img_ids = df['ID']
  
    xCnt = 128
    yCnt = 128
    imgs = []
    
    for img_name in img_ids:
        # read in as grey img [0, 1]
        img = skio.imread('%s/%s.jpg' % (data_dir, img_name), as_gray=True)  
        height, width = np.shape(img)[0:2] 
        img = skt.resize(img, (INPUT_HEIGHT, INPUT_WIDTH)) 
        imgs.append(img)

    fScale = INPUT_HEIGHT*1.0/height
    if load_pts:
        # pts are not normalized
        x1 = np.array(df['X1'].values * fScale)
        y1 = np.array(df['Y1'].values * fScale)
        x2 = np.array(df['X2'].values * fScale)
        y2 = np.array(df['Y2'].values * fScale)
        pts1 = np.stack((x1, y1), axis=1)
        pts2 = np.stack((x2, y2), axis=1)
        #pts0 = np.stack((pts1, pts2), axis=1)

    print('Num of images: {}'.format(len(imgs)))

    if load_pts:
        return img_ids, imgs, pts1, pts2
    else:
        return img_ids, imgs


data_dir='./data/testing/frames'
data_csv='./data/testing/ch234.csv'
_, imgs, pts1, pts2 = load_data(data_dir, data_csv)
pts01 = np.append(pts1, pts2, axis=1)
testData = np.asarray(imgs)
testLabel= torch.tensor(np.asarray(pts01))



def show_img(img, pts1, pts2):
    import sys
    import matplotlib.pyplot as plt
    #import matplotlib.patches as patches

    def press(event):
        if event.key == 'q':
            print('Terminated by user')
            sys.exit()
        elif event.key == 'c':
            plt.close()


    x1, y1 = pts1
    x2, y2 = pts2

    plt.ioff()
    fig = plt.figure(frameon=False)
    #fig.canvas.set_window_title('Image')
    fig.canvas.mpl_connect('key_press_event', press)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(img, cmap='gray')
    ax.plot([x1, x2], [y1, y2], 'r-', lw=2)
    print(x1, y1, x2, y2)
    plt.scatter([x1, x2], [y1, y2], s=20)
    plt.show()


class TestDataset(Dataset):
    def __init__(self, input, label):
        super(Dataset).__init__()
        self.input = input
        self.label = label

    def __getitem__(self, item):
        #input = self.input[item,:,:]
        #label = self.label[item,:]

        return self.input[item,:,:], self.label[item,:]

    def __len__(self):
        return self.input.shape[0]


test_iter = \
    DataLoader(TestDataset(torch.tensor(testData).float(), testLabel.float()),
                    batch_size=BATCH_SIZE,
                    shuffle=False)



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # The first Convolution layer (1,20) 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(in_features=32768, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.fc3 = nn.Linear(in_features=500, out_features=OUTPUTS)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.5)
        self.dropout5 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        input = self.conv1(x)
        input = self.bn1(input)
        input = self.relu(input)
        input = self.pool_1(input)
        input = self.dropout1(input)

        input = self.conv2(input)
        input = self.bn2(input)
        input = self.relu(input)
        input = self.pool_2(input)
        input = self.dropout2(input)

        input = self.conv3(input)
        input = self.bn3(input)
        input = self.relu(input)
        input = self.pool_3(input)
        input = self.dropout3(input)
        input = input.flatten(start_dim=1)

        input = self.fc1(input)
        input = self.relu(input)
        input = self.dropout4(input)
        input = self.fc2(input)
        input = self.relu(input)
        input = self.dropout5(input)
        input = self.fc3(input)
        output = self.relu(input)


        return output


net = torch.load('./best_model.pkl')


if torch.cuda.is_available():
    net.cuda()
    
loss_function = nn.MSELoss()
loss_function_test = nn.L1Loss()


total_loss = 0.0
num = 0
loss_array = []


for X, y in test_iter:
    if torch.cuda.is_available():
        X = X.to(device)
        y = y.to(device)

    y_hat = net(X.float())
    num = num + 1
    loss = loss_function_test(y_hat, y).item()
    total_loss += loss
    loss_array.append(loss)

    #visualization
    #for k in range(X.shape[0]):
    #    show_img(X[k].cpu(), y[k][0:2].cpu(), y[k][2:4].cpu())
    #    show_img(X[k].cpu(), y_hat[k][0:2].cpu().detach().numpy(), y_hat[k][2:4].cpu().detach().numpy())
    #break
    
    
print('average L1 loss = ', total_loss/num)

import numpy as np
print("average = ", np.mean(loss_array), " minimum = ", np.min(loss_array), " maximum = ", np.max(loss_array)) 





