import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data as Data
from Functions import Predict_Dataset_epoch
import pandas as pd
import torch.nn as nn

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


class memory_conv(nn.Module):
    def __init__(self):
        super(memory_conv, self).__init__()
        memory_tensor = torch.rand((16, 1, 256, 256))
        self.memory_img = torch.nn.Parameter(memory_tensor, requires_grad=True)
        self.conv7x7 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        batch_num = x.shape[0]
        cat_memory = torch.cat([x, self.memory_img[:batch_num]], dim=1)
        result = self.conv7x7(cat_memory)
        return result


# Read test data
df = pd.read_csv('../Data/submission_sample.csv')
# print(df.values)
name_list = df.values[:, 0]
test_data_path = '../Data/test/test_contest/test/'

for index in range(len(name_list)):
    name_list[index] = test_data_path + name_list[index] +'.bmp'

print(name_list)

result_list = [['id', 'defect_score']]
# result_list.append(['WEP', 0.7])

# np.savetxt("test.csv", result_list, delimiter=',', fmt='%s')

valid_generator = Data.DataLoader(Predict_Dataset_epoch(name_list), batch_size=16, shuffle=False, num_workers=4)

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.conv1 = memory_conv()
model.fc = torch.nn.Linear(2048, 2)
model.to(device)
print(model)

print()
print("Loading model...")
model_path = '../Model/resnet50_memory_subset_crossentro_0050.pth'
print(model_path)
model.load_state_dict(torch.load(model_path))

with torch.no_grad():
    for X_image, X_name in valid_generator:
        X_image = X_image.to(device)

        predict = model(X_image)

        predict_argmax = torch.argmax(predict, dim=1)

        predict_argmax_npy = predict_argmax.data.cpu().numpy()

        for index in range(len(X_name)):
            target_name = X_name[index].split('/')[-1].split('.')[0]
            result_list.append([target_name, predict_argmax_npy[index].item()])


np.savetxt("resnet50_memory_subset_crossentro_0050.csv", result_list, delimiter=',', fmt='%s')
print("done")