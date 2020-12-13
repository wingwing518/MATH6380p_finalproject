# import SimpleITK as sitk
import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import glob
import itertools
from scipy.ndimage.interpolation import zoom as zoom
import random
# from scipy.ndimage import shift, rotate
import csv
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from matplotlib.pyplot import imshow

def min_max_norm(img):
    img_max = np.max(img)
    img_min = np.min(img)
    return (img-img_min)/(img_max-img_min)

def Norm_Zscore(img):
    img = (img - torch.mean(img)) / torch.std(img)
    return img


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, labels, norm=False):
        super(Dataset_epoch, self).__init__()
        'Initialization'
        self.names = names
        self.norm = norm
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img = Image.open(self.names[step])
        img = img.convert('RGB')
        # img.show()
        # print(np.asarray(img).shape)
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        label = self.labels[step]

        img_tensor = preprocess(img)
        label_tensor = torch.from_numpy(label).float()

        if self.norm:
            return Norm_Zscore(img_tensor), label_tensor
        else:
            return img_tensor.float(), label_tensor


class Dataset_epoch_train(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, labels, norm=False):
        super(Dataset_epoch_train, self).__init__()
        'Initialization'
        self.names = names
        self.norm = norm
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img = Image.open(self.names[step])
        img = img.convert('RGB')
        # img.show()
        # print(np.asarray(img).shape)
        preprocess = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        label = self.labels[step]

        img_tensor = preprocess(img)
        label_tensor = torch.from_numpy(label).float()

        if self.norm:
            return Norm_Zscore(img_tensor), label_tensor
        else:
            return img_tensor.float(), label_tensor


class Balance_dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, pos_names, neg_names, norm=False):
        super(Balance_dataset_epoch, self).__init__()
        'Initialization'
        self.pos_names = pos_names
        self.neg_names = neg_names
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pos_names) + len(self.neg_names)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        random_flag = random.randint(0, 1)
        # print(random_flag)

        if random_flag == 0:
            index = step % len(self.pos_names)
            img = Image.open(self.pos_names[index])
            label = np.array([1., 0])
        else:
            index = step % len(self.neg_names)
            img = Image.open(self.neg_names[index])
            label = np.array([0, 1.])

        img = img.convert('RGB')
        # img.show()
        # print(np.asarray(img).shape)
        preprocess = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # label = self.labels[step]

        img_tensor = preprocess(img)
        label_tensor = torch.from_numpy(label).float()

        if self.norm:
            return Norm_Zscore(img_tensor), label_tensor
        else:
            return img_tensor.float(), label_tensor


class Predict_Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=False):
        super(Predict_Dataset_epoch, self).__init__()
        'Initialization'
        self.names = names
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img = Image.open(self.names[step])
        img = img.convert('RGB')
        # img.show()
        # print(np.asarray(img).shape)
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = preprocess(img)


        if self.norm:
            return Norm_Zscore(img_tensor), self.names[step]
        else:
            return img_tensor.float(), self.names[step]


if __name__ == '__main__':
    train_path = '../Data/train/train_contest'
    train_pos_list = sorted(glob.glob(train_path + '/good_all/*.bmp'))
    train_neg_list = sorted(glob.glob(train_path + '/defect/*.bmp'))

    data_gen = Balance_dataset_epoch(train_pos_list, train_neg_list)
    data_gen.__getitem__(0)