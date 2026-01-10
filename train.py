import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import time
import datetime
import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader
# from utils import print_and_save, shuffling, epoch_time
# from network.Mart_Unet import MartingaleUNet
from network.lianxi import MedSegNet

from metrics import DiceBCELoss
from run_engine import *
from sklearn.utils import shuffle

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from utils.run_model import load_data, train, evaluate, DATASET, CompositeLoss, FocalLoss, DiceLoss, combined_loss
from sklearn.metrics import accuracy_score
import torch.backends.cudnn as cudnn
import cv2
import numpy as np

cudnn.enabled = True
cudnn.benchmark = False  # 禁用自动最优算法搜索
cudnn.deterministic = True  # 强制使用确定性算法


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


""" Shuffle the dataset. """


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")


if __name__ == '__main__':
    # 训练集和验证集
    dataset_name = 'ISIC2016'
    val_name = 'Val_Images'

    # 定义随机种子
    seed = random.randint(0, 10000)
    my_seeding(seed)

    # 训练相关参数
    image_size = 256
    size = (image_size, image_size)
    batch_size = 4
    num_epochs = 200
    lr = 1e-4
    early_stopping_patience = 100

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"train_{dataset_name}_{val_name}_lr{lr}_{current_time}"

    # 输入路径和权重保存路径设置
    base_dir = r".\data"
    # data_path = os.path.join(base_dir, dataset_name)
    data_path = os.path.join(".", "data", "ISIC2017")
    save_dir = os.path.join("073run_files", dataset_name, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 训练日志和checkpoint_path保存
    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    train_log = open(train_log_path, 'w')
    train_log.write("\n")
    train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    # 一些超参数
    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Seed: {seed}\n"
    print_and_save(train_log_path, hyperparameters_str)

    """ Data augmentation: Transforms """

    transform = A.Compose([
        A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader"""
    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, val_name)
    num_with_fg = 0
    # for mask_path in train_y:
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #     mask_bin = (mask > 127).astype(np.float32)
    #     if np.sum(mask_bin) > 0:
    #         num_with_fg += 1
    # for mask_path in train_y:
    #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #     mask_bin = (mask > 127).astype(np.float32)
    #     print("[DEBUG] mask unique:", np.unique(mask_bin), "shape:", mask.shape)
    #
    # print(f"[DEBUG] 有前景的mask数: {num_with_fg}, 总图片数: {len(train_y)}")
    # print(f"[DEBUG] 前景比例: {num_with_fg / len(train_y):.4f}")
    # import matplotlib.pyplot as plt
    # mask = cv2.imread(train_y[0], cv2.IMREAD_GRAYSCALE)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    # print('mask unique:', np.unique(mask))

    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    # for img, mask in train_loader:
    #     print("图像shape:", img.shape, "像素范围:", img.min().item(), img.max().item())
    #     print("标签unique:", torch.unique(mask))
    #     break

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ Model """
    device = torch.device('cuda')
    # 先暂时这么写，后续再改
    model = MedSegNet()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, verbose=True)
    # loss_fn = CompositeLoss(
    #     alpha=1.0,  # BCE 权重
    #     beta=1.0,  # Dice 权重
    #     # gamma=0.2,  # Texture loss 权重
    #     # in_channels=1,  # 输入通道数（通常是分割 mask 通道数）
    #     # theta=1.0  # 鞅网络的超参数
    # )
    # loss_name = "CompositeLoss"

    dice_loss_fn = DiceLoss()
    focal_loss_fn = FocalLoss()

    # loss_fn = combined_loss(pred, y)
    # loss_name = "combined_loss"
    dice_loss_fn = DiceLoss()
    focal_loss_fn = FocalLoss()


    def combined_loss(logits, targets):
        return 0.5 * dice_loss_fn(logits, targets) + 0.5 * focal_loss_fn(logits, targets)


    loss_fn = combined_loss  # 注意这里没有括号！
    loss_name = "combined_loss"

    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[0] > best_valid_metrics:
            data_str = f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0


        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        with open(os.path.join(save_dir, "train_log.csv"), "a") as f:
            f.write(
                f"{epoch + 1},{train_loss},{train_metrics[0]},{train_metrics[1]},{train_metrics[2]},{train_metrics[3]},{valid_loss},{valid_metrics[0]},{valid_metrics[1]},{valid_metrics[2]},{valid_metrics[3]}\n")

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
