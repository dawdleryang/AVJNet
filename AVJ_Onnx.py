#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn


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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 128, 128)
input_names = [ "AVJNet_input" ]
output_names = [ "AVJNet_output" ]
net = torch.load('./AVJNet_model.pkl')
net.eval()

torch.onnx.export(net,
                 dummy_input.to(device) ,
                 "AVJNet.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 export_params=True,
                 )




