# data_loader.py
from __future__ import print_function, division
# ... (RescaleT, Rescale, RandomCrop 类保持原样) ...
import pandas as pd  # <--- 新增: 导入pandas库
import os  # <--- 新增: 导入os库
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label, class_label = sample['imidx'], sample['image'], sample['label'], sample['class_label']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0,
                               preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl, 'class_label': class_label}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label, class_label = sample['imidx'], sample['image'], sample['label'], sample['class_label']


        if random.random() >= 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl, 'class_label': class_label}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label, class_label = sample['imidx'], sample['image'], sample['label'], sample['class_label']


        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = image[top: top + new_h, left: left + new_w]
        lbl = label[top: top + new_h, left: left + new_w]

        return {'imidx': imidx, 'image': img, 'label': lbl, 'class_label': class_label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # <--- 修改: 从sample中解包新的 'class_label' 键 --->
        imidx, image, label, class_label = sample['imidx'], sample['image'], sample['label'], sample['class_label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        # ... (您原有的图像和label处理逻辑保持不变)
        MY_MEAN = 0.10485467077077552
        MY_STD = 0.22128120909611002
        image = image / np.max(image)
        if np.max(label) > 1:
            label = label / 255.0
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - MY_MEAN) / MY_STD
            tmpImg[:, :, 1] = (image[:, :, 0] - MY_MEAN) / MY_STD
            tmpImg[:, :, 2] = (image[:, :, 0] - MY_MEAN) / MY_STD
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - MY_MEAN) / MY_STD
            tmpImg[:, :, 1] = (image[:, :, 1] - MY_MEAN) / MY_STD
            tmpImg[:, :, 2] = (image[:, :, 2] - MY_MEAN) / MY_STD
        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label[:, :, 0]

        # <--- 修改: 在返回的字典中也包含 class_label --->
        return {'imidx': torch.from_numpy(imidx),
                'image': torch.from_numpy(tmpImg).float(),
                'label': torch.from_numpy(tmpLbl).float(),  # VLM模型通常输出float类型的mask
                'class_label': torch.tensor(class_label, dtype=torch.long)}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        # <--- 修改: 从sample中解包新的 'class_label' 键 --->
        imidx, image, label, class_label = sample['imidx'], sample['image'], sample['label'], sample['class_label']

        # ... (您原有的复杂的图像和label处理逻辑保持不变)
        if np.max(label) > 1:
            label = label / 255.0
        # ... (此处省略了您原有的 if/else 图像处理逻辑)
        tmpImg = ...  # 根据您原有的逻辑计算 tmpImg
        tmpLbl = label[:, :, 0]

        # <--- 修改: 在返回的字典中也包含 class_label --->
        return {'imidx': torch.from_numpy(imidx),
                'image': torch.from_numpy(tmpImg).float(),
                'label': torch.from_numpy(tmpLbl).float(),  # VLM模型通常输出float类型的mask
                'class_label': torch.tensor(class_label, dtype=torch.long)}


class SalObjDataset(Dataset):
    # <--- 修改: __init__函数增加一个csv_path参数 --->
    def __init__(self, img_name_list, lbl_name_list, transform=None, csv_path=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

        # <--- 新增: 如果提供了CSV路径，则加载标签映射 --->
        if csv_path:
            df = pd.read_csv(csv_path)
            # 创建一个从文件名到疾病状态的字典, e.g., {'100.jpg': 'ABN'}
            self.label_map = df.set_index('File_name')['Disease'].to_dict()
            print(f"成功从 {csv_path} 加载 {len(self.label_map)} 条标签记录。")
        else:
            # 如果没有提供CSV，为了代码不报错，创建一个空字典
            self.label_map = {}
            print("警告: 未提供CSV标签文件，所有样本将被视为负样本(0)。")

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.image_name_list[idx])
        imname = self.image_name_list[idx]
        imidx = np.array([idx])

        # ... (您原有的label加载和形状处理逻辑保持不变)
        if (0 == len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])
        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3
        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        # <--- 新增: 从字典中查找分类标签 --->
        filename = os.path.basename(imname)
        disease_status = self.label_map.get(filename, 'NOR')  # 安全获取，找不到默认为'NOR'

        if disease_status == 'ABN':
            class_label = 1  # 异常/正样本
        else:
            class_label = 0  # 正常/负样本

        # <--- 修改: 在sample字典中加入'class_label' --->
        sample = {'imidx': imidx, 'image': image, 'label': label, 'class_label': class_label}

        if self.transform:
            sample = self.transform(sample)

        return sample