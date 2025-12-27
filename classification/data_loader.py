# dataloader.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


# ================================================================
# ==== 1. 数据转换类 (保持您原有的设计) ====
# ================================================================

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

class DataTransforms:
    """
    管理训练和验证数据转换的类。
    """

    def __init__(self, output_size=224):
        dataset_mean = [0.1048, 0.1048, 0.1048]  # 示例：ImageNet 均值
        dataset_std = [0.2212, 0.2212, 0.2212]  # 示例：ImageNet 标准差

        self.train = transforms.Compose([
            transforms.RandomResizedCrop(output_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 增加颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ])

        self.val = transforms.Compose([
            transforms.Resize(output_size + 32),
            transforms.CenterCrop(output_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ])


# ================================================================
# ==== 2. 原有的标准分类 Dataset (保持不变) ====
# ================================================================

class DiseaseClassificationDataset(Dataset):
    """
    用于疾病二分类任务的数据集类。
    从 CSV 文件读取文件名和标签。
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.annotations_frame['label'] = self.annotations_frame['Disease'].apply(lambda x: 0 if x == 'NOR' else 1)

    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations_frame.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"警告：找不到文件 {img_name}，将跳过。")
            return None, None  # 返回None以便collate_fn处理

        label = self.annotations_frame.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# =======================================================================
# ==== 3.【新功能】集成了智能裁剪功能的分类 Dataset ====
# =======================================================================

class CroppedClassificationDataset(Dataset):
    """
    一个特殊的分类数据集：
    在加载时，会使用对应的mask文件，先将原图裁剪到目标区域，然后再进行数据变换。
    """

    def __init__(self, csv_file, root_dir, mask_dir, transform=None, padding=10):
        """
        Args:
            csv_file (string): 包含标注信息的 CSV 文件路径。
            root_dir (string): 包含所有图像的目录路径。
            mask_dir (string): 包含所有分割掩码的目录路径。
            transform (callable, optional): 应用于样本的可选变换。
            padding (int): 裁剪边界的额外留白。
        """
        self.annotations_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.padding = padding
        self.annotations_frame['label'] = self.annotations_frame['Disease'].apply(lambda x: 0 if x == 'NOR' else 1)

    def __len__(self):
        return len(self.annotations_frame)

    def _get_square_crop_box(self, mask_np):
        """根据numpy格式的掩码计算正方形裁剪坐标"""
        rows, cols = np.where(mask_np > 128)
        if len(rows) == 0:
            raise ValueError("掩码中未找到目标区域。")

        x_min, x_max, y_min, y_max = np.min(cols), np.max(cols), np.min(rows), np.max(rows)
        side_length = max(x_max - x_min, y_max - y_min)

        center_x = x_min + (x_max - x_min) / 2
        center_y = y_min + (y_max - y_min) / 2

        new_x_min = int(center_x - side_length / 2 - self.padding)
        new_y_min = int(center_y - side_length / 2 - self.padding)
        new_x_max = new_x_min + side_length + self.padding * 2
        new_y_max = new_y_min + side_length + self.padding * 2

        img_h, img_w = mask_np.shape
        return max(0, new_x_min), max(0, new_y_min), min(img_w, new_x_max), min(img_h, new_y_max)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. 获取图像路径和标签
        image_filename = self.annotations_frame.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, image_filename)
        label = self.annotations_frame.iloc[idx, -1]

        try:
            image_pil = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"警告：找不到图像文件 {img_path}，将跳过。")
            return None, None

        # 2. 推断并加载对应的掩码
        base_name = os.path.splitext(image_filename)[0]
        mask_path = os.path.join(self.mask_dir, base_name + '.png')  # 假设掩码都是png格式

        image_to_transform = image_pil  # 默认使用原图

        try:
            mask_pil = Image.open(mask_path).convert('L')
            mask_np = np.array(mask_pil)

            # 3. 计算裁剪坐标并裁剪
            x1, y1, x2, y2 = self._get_square_crop_box(mask_np)
            image_to_transform = image_pil.crop((x1, y1, x2, y2))

        except FileNotFoundError:
            # print(f"信息：找不到掩码 {mask_path}，将使用原图进行分类。")
            pass  # 如果找不到mask，就静默地使用原图
        except ValueError:
            # print(f"信息：掩码 {mask_path} 为空，将使用原图进行分类。")
            pass  # 如果mask是全黑的，也静默地使用原图

        # 4. 对裁剪后（或原始）的图像应用变换
        if self.transform:
            final_image = self.transform(image_to_transform)

        return final_image, torch.tensor(label, dtype=torch.long)