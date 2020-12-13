import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data as Data
from Functions import Predict_Dataset_epoch
import pandas as pd

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

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

valid_generator = Data.DataLoader(Predict_Dataset_epoch(name_list), batch_size=32, shuffle=False, num_workers=4)

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
model.fc = torch.nn.Linear(2048, 2)

# model.fc = torch.nn.Sequential(
#     torch.nn.Linear(in_features=2048, out_features=2, bias=True),
#     torch.nn.Sigmoid()
# )
model.to(device)
print(model)

print()
print("Loading model...")
model_path = '../Selected_model/resnet50_2_subset_crossentro_0050.pth'
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


np.savetxt("resnet50_2_subset_crossentro_0050.csv", result_list, delimiter=',', fmt='%s')
print("done")