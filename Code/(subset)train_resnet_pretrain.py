import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data as Data
from Functions import Dataset_epoch, Balance_dataset_epoch, Dataset_epoch_train
from torch.optim import lr_scheduler
import pandas as pd

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

# Read data path
# train_path = '../Data/train/train_contest'
# test_path = '../Data/test'
#
# train_pos_list = sorted(glob.glob(train_path + '/good_all/*.bmp'))
# train_neg_list = sorted(glob.glob(train_path + '/defect/*.bmp'))
#
# # for validate set
# train_pos_label = np.zeros((len(train_pos_list), 2))
# train_pos_label[:, 0] = 1.
# train_neg_label = np.zeros((len(train_neg_list), 2))
# train_neg_label[:, 1] = 1.
#
# merge_train_list = train_pos_list[0:500] + train_neg_list[0:500]
# merge_label_list = np.vstack((train_pos_label[0:500], train_neg_label[0:500]))

train_data = pd.read_csv('../Data/train_list.csv')
val_data = pd.read_csv('../Data/train_val_list.csv')

print('Training data size: ', len(train_data))
print('Validation data size: ', len(val_data))


def load_data(data):
    x = []
    y = []

    for index, id in enumerate(data['id']):
        image_id = id
        if data['label'][index] == 0:
            label = 'good_all'
            image_path = '../Data/train/train_contest/' + label + '/' + image_id
            x.append(image_path)
            y.append([1., 0])
        else:
            label = 'defect'
            image_path = '../Data/train/train_contest/' + label + '/' + image_id
            x.append(image_path)
            y.append([0, 1.])
        #image = np.array(image, dtype=np.uint8)
        #plt.imshow(image)
        #plt.gca().axis('off')
    y = np.array(y)
    return x, y


train_x, train_y = load_data(train_data)
val_x, val_y = load_data(val_data)

# Dataloader
# training_generator = Data.DataLoader(Balance_dataset_epoch(train_pos_list, train_neg_list), batch_size=16,
#                                      shuffle=True, num_workers=4)
training_generator = Data.DataLoader(Dataset_epoch_train(train_x, train_y), batch_size=16,
                                     shuffle=True, num_workers=4)
valid_generator = Data.DataLoader(Dataset_epoch(val_x, val_y), batch_size=64,
                                     shuffle=False, num_workers=4)

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
print(model)
model.fc = torch.nn.Linear(2048, 2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
# loss_mse = torch.nn.MSELoss()
loss_mse = torch.nn.CrossEntropyLoss()

max_epoch = 50
step = 0
epoch_index = 0
model_path = '../Model/'
accuracy_list = []

while epoch_index <= max_epoch:
    for X, X_label in training_generator:
        X = X.to(device)
        X_label = X_label.to(device, dtype=torch.long)
        predict = model(X)

        # Backpropagation
        # loss = loss_mse(predict, X_label)
        loss = loss_mse(predict, torch.max(X_label, 1)[1])

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        sys.stdout.write(
            "\r" + 'step "{0}" -> training loss "{1:.4f}"'.format(
                step, loss.item()))
        sys.stdout.flush()

        step += 1

    epoch_index += 1
    # learning rate decay
    exp_lr_scheduler.step()
    # Save model
    if epoch_index % 2 == 0 or epoch_index == 1:
        print()
        print("Validating...")
        modelname = model_path + "resnet50_2_subset_crossentro_" + str(epoch_index).zfill(4) + '.pth'
        torch.save(model.state_dict(), modelname)

        num_correct = 0
        num_data = len(val_x)
        with torch.no_grad():
            for X_image, X_label in valid_generator:
                X_image, X_label = X_image.to(device), X_label.to(device)

                predict = model(X_image)

                predict_argmax = torch.argmax(predict, dim=1)
                X_label_argmax = torch.argmax(X_label, dim=1)

                # accuracy
                num_correct += (predict_argmax == X_label_argmax).sum().item()

            accuracy = num_correct/num_data
            accuracy_list.append(accuracy)
            print("Epoch_index:", str(epoch_index).zfill(4), "Accuracy: ", str(accuracy))
            print()


accuracy_list_npy = np.array(accuracy_list)
print(accuracy_list_npy.shape)
np.save("../Log/resnet50_2_subset_accuracy_list.npy", accuracy_list_npy)
print()