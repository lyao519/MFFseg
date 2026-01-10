import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

# from utils import calculate_metrics
from tqdm import tqdm
from sklearn.metrics import accuracy_score

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F


def Train_load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path, "Images", name) + ".jpg" for name in data]
    masks = [os.path.join(path, "Masks", name) + ".jpg" for name in data]
    # images = [os.path.join(path, "images", name) + ".png" for name in data]
    # masks = [os.path.join(path, "masks", name) + ".png" for name in data]
    return images, masks


def Val_load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path, "Val_Images", name) + ".jpg" for name in data]
    masks = [os.path.join(path, "Val_Masks", name) + ".jpg" for name in data]
    # images = [os.path.join(path, "images", name) + ".png" for name in data]
    # masks = [os.path.join(path, "masks", name) + ".png" for name in data]
    return images, masks


def load_data(path, val_name=None):
    train_names_path = f"{path}/train.txt"
    # valid_names_path = f"{path}/val.txt"
    if val_name is None:
        valid_names_path = f"{path}/val.txt"
    else:
        valid_names_path = f"{path}/val.txt"

    train_x, train_y = Train_load_names(path, train_names_path)
    valid_x, valid_y = Val_load_names(path, valid_names_path)  # 读取图像和掩码的路径

    return (train_x, train_y), (valid_x, valid_y)


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        # print("n_samples:", self.n_samples)
        # self.convert_edge=convert_edge
        self.size = size

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        background = mask.copy()
        background = 255 - background

        """ Applying Data Augmentation """
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, background=background)
            image = augmentations["image"]
            mask = augmentations["mask"]
            background = augmentations["background"]

        """ Image """
        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        """ Mask """
        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        """ Background """
        background = cv2.resize(background, self.size)
        background = np.expand_dims(background, axis=0)
        background = background / 255.0

        return image, (mask, background)

    def __len__(self):
        return self.n_samples


def load_test_data(path):
    """
    加载测试集图像和标签路径。

    参数:
        path (str): 测试集根路径，包含 'images' 和 'masks' 文件夹

    返回:
        test_x (list): 图像路径列表
        test_y (list): 标签路径列表
    """
    image_dir = os.path.join(path, 'Test_Images')
    mask_dir = os.path.join(path, 'Test_Masks')

    # 检查路径
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图像路径不存在: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"标签路径不存在: {mask_dir}")

    image_list = sorted(os.listdir(image_dir))
    mask_list = sorted(os.listdir(mask_dir))

    test_x = []
    test_y = []

    for img_name in image_list:
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)  # 假设标签与图像同名

        if not os.path.isfile(mask_path):
            print(f"警告: 缺失标签 {mask_path}")
            continue

        test_x.append(img_path)
        test_y.append(mask_path)

    print(f"共加载 {len(test_x)} 张图像路径和标签路径。")
    return test_x, test_y
