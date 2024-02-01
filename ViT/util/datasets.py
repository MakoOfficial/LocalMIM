# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import Tensor

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
from PIL import Image
import numpy as np

class BAATrainDataset(Dataset):
    def __init__(self, df_path, file_path, transform):
        def preprocess_df(df_path):
            # nomalize boneage distribution
            # df['zscore'] = df['boneage'].map(lambda x: (x - boneage_mean) / boneage_div)
            # change the type of gender, change bool variable to float32
            df = pd.read_csv(df_path)
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df_path)
        self.file_path = file_path
        self.transform = transform
    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        # return (transform_train(image=read_image(f"{self.file_path}/{num}.png"))['image'],
        #         Tensor([row['male']])), row['zscore']
        img = cv2.imread(f"{self.file_path}/{num}.png", cv2.IMREAD_COLOR)
        img = Image.fromarray(np.uint8(img))
        return (self.transform(img), Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)


class BAAValDataset(Dataset):
    def __init__(self, df_path, file_path, transform):
        def preprocess_df(df_path):
            # change the type of gender, change bool variable to float32
            df = pd.read_csv(df_path)
            df['male'] = df['male'].astype('float32')
            df['bonage'] = df['boneage'].astype('float32')
            return df

        self.df = preprocess_df(df_path)
        self.file_path = file_path
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = cv2.imread(f"{self.file_path}/{int(row['id'])}.png", cv2.IMREAD_COLOR)
        img = Image.fromarray(np.uint8(img))
        return (self.transform(img), Tensor([row['male']])), row['boneage']

    def __len__(self):
        return len(self.df)


def build_dataset_BAA(is_train, args):
    transform = build_transform(is_train, args)
    # root = os.path.join(args.data_path, 'train' if is_train else 'valid')
    if is_train:
        dataset = BAATrainDataset(args.train_csv, args.train_path, transform)
    else:
        dataset = BAAValDataset(args.valid_csv, args.valid_path, transform)
    print(dataset)
    return dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=transforms.InterpolationMode.BICUBIC,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std)
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224/256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))  # to maintain same ratio w.r.t. 224 images
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)