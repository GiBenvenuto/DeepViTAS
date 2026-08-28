import torch
import numpy as np
import random
import os
import cv2
from tqdm import tqdm as tqdm
from pandas import read_csv
from math import ceil
from torch.utils.data import DataLoader, Dataset
from skimage import io



class RandomFlip:
    def __init__(self, generator=None):
        self.generator = generator

    def __call__(self, sample):
        I1, label = sample['I1'], sample['label']
        if torch.rand(1, generator=self.generator).item() > 0.5:
            I1 = torch.flip(I1, dims=[2])
            label = torch.flip(label, dims=[1])
        return {'I1': I1, 'label': label}


class RandomRot:
    def __init__(self, generator=None):
        self.generator = generator

    def __call__(self, sample):
        I1, label = sample['I1'], sample['label']
        n = torch.randint(0, 4, (1,), generator=self.generator).item()
        if n:
            I1 = torch.rot90(I1, n, dims=(1, 2))
            label = torch.rot90(label, n, dims=(0, 1))
        return {'I1': I1, 'label': label}

def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    #     out = np.swapaxes(I,1,2)
    #     out = np.swapaxes(out,0,1)
    #     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)

def read_sentinel_img_diff2(path):
    """Read cropped Sentinel-2 image: RGB bands."""
    bitwise = cv2.imread(path + "/pair/img_diff.png")
    bitwise = cv2.cvtColor(bitwise, cv2.COLOR_RGB2GRAY)
    img1 = cv2.imread(path + "/pair2/img1.png")
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.imread(path + "/pair2/img2.png")
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    I = np.stack((img1, img2, bitwise), axis=-1).astype('float')

    I = (I - I.mean()) / I.std()

    return I







def read_sentinel_img_trio_diff(path, name, TYPE=0):
    """Read cropped Sentinel-2 image pair and change map."""
    #     read images
    if TYPE == 0:
        # print(path)
        I1 = read_sentinel_img_diff2(path + name)
    elif TYPE == 1:
        print("Not implemented :(")
        exit(0)

    cm = io.imread(path + name + '/cm/cm_gt.png', as_gray=True) != 0

    return I1, cm

class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train=0, patch_side=96, stride=None, transform=None, TYPE=0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # basics
        self.transform = transform
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride

        if train == 0:
            fname = 'train.txt'
        elif train == 1:
            fname = 'val.txt'
        else:
            fname = 'test.txt'

        print(path + fname)
        self.names = read_csv(path + fname).columns
        self.n_imgs = self.names.shape[0]

        n_pix = 0
        true_pix = 0

        # load images
        self.imgs_1 = {}
        # self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image

            I1, cm = read_sentinel_img_trio_diff(self.path, im_name, TYPE)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            # self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm

            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()

            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i

            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (im_name,
                                            [self.stride * i, self.stride * i + self.patch_side, self.stride * j,
                                             self.stride * j + self.patch_side],
                                            [self.stride * (i + 1), self.stride * (j + 1)])
                    self.patch_coords.append(current_patch_coords)

        self.weights = [10 * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]

        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        # I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]

        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1 * np.array(label)).float()

        sample = {'I1': I1, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample