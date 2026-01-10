import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import cv2
from tqdm import tqdm
import torch
# 导入我们的模型
from network.model import ConDSeg

from utils import create_dir, seeding
from utils import calculate_metrics
from run_engine import load_data
from run_engine import load_test_data


def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def process_edge(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)

    y_pred = y_pred > 0.001
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def print_score(metrics_score):
    jaccard = metrics_score[0] / len(test_x)  #
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    f2 = metrics_score[5] / len(test_x)

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")


def evaluate(model, save_path, test_x, test_y, size):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)

        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            mask_pred, fg_pred, bg_pred, uc_pred = model(image)
            p1 = mask_pred

            """ Evaluation metrics """
            score_1 = calculate_metrics(mask, p1)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))
            p1 = process_mask(p1)

        # 构造保存目录并确保存在
        mask_save_dir = os.path.join(save_path, "mask")
        os.makedirs(mask_save_dir, exist_ok=True)

        # 保存预测掩膜
        output_path = os.path.join(mask_save_dir, f"{name}.jpg")
        cv2.imwrite(output_path, p1)

    print_score(metrics_score_1)

    with open(f"{save_path}/result.txt", "w") as file:
        file.write(f"Jaccard: {metrics_score_1[0] / len(test_x):1.4f}\n")
        file.write(f"F1: {metrics_score_1[1] / len(test_x):1.4f}\n")
        file.write(f"Recall: {metrics_score_1[2] / len(test_x):1.4f}\n")
        file.write(f"Precision: {metrics_score_1[3] / len(test_x):1.4f}\n")
        file.write(f"Acc: {metrics_score_1[4] / len(test_x):1.4f}\n")
        file.write(f"F2: {metrics_score_1[5] / len(test_x):1.4f}\n")


if __name__ == '__main__':
    """ Seeding """
    dataset_name = 'ISIC2018'
    seeding(42)
    size = (256, 256)

    # 载入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConDSeg(256, 256)
    model = model.to(device)
    checkpoint_path = r".\run_files\ISIC2018\ISIC2018_Val_Images_lr1e-05_20250602-142258\checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 载入测试集
    path = r".\data\{}".format(dataset_name)
    (test_x, test_y) = load_test_data(path)

    # 保存路径
    save_path = f"results/{dataset_name}/MyModel"
    create_dir(f"{save_path}/mask")

    # 初始化指标累加器
    total_metrics = np.zeros(6)  # [Jaccard, F1, Recall, Precision, Acc, F2]
    total_iou = 0.0  # IoU 累加
    total_dice = 0.0  # Dice 累加
    total_count = len(test_x)

    for x_path, y_path in tqdm(zip(test_x, test_y), total=total_count):
        # 读取图像
        image = cv2.imread(x_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        img_input = np.transpose(image, (2, 0, 1)) / 255.0
        img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
        img_input = torch.from_numpy(img_input).to(device)

        # 读取掩码
        mask = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        mask_input = np.expand_dims(mask, axis=0) / 255.0
        mask_input = np.expand_dims(mask_input, axis=0).astype(np.float32)
        mask_input = torch.from_numpy(mask_input).to(device)

        # 预测
        with torch.no_grad():
            pred_mask, _, _, _ = model(img_input)

        # 后处理预测结果（可视化掩码）
        vis_mask = process_mask(pred_mask)

        # 保存预测图
        name = os.path.basename(y_path).split('.')[0]
        mask_save_path = os.path.join(save_path, "mask")
        os.makedirs(mask_save_path, exist_ok=True)
        cv2.imwrite(os.path.join(mask_save_path, f"{name}.jpg"), vis_mask)

        # 计算标准指标
        metrics = calculate_metrics(mask_input, pred_mask)[:6]
        total_metrics += np.array(metrics)

        # IoU 和 Dice
        pred_bin = (pred_mask[0].cpu().numpy() > 0.5).astype(np.uint8)
        gt_bin = mask_input.cpu().numpy().squeeze(0).squeeze(0).astype(np.uint8)

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()
        total_iou += intersection / union if union > 0 else 0.0

        dice = (2. * intersection) / (pred_bin.sum() + gt_bin.sum()) if (pred_bin.sum() + gt_bin.sum()) > 0 else 0.0
        total_dice += dice

        # 计算均值
    avg_metrics = total_metrics / total_count
    avg_iou = total_iou / total_count
    avg_dice = total_dice / total_count

    # 输出结果
    print(f"\n【测试集平均指标】")
    print(f"Jaccard (IoU): {avg_metrics[0]:.4f}")
    print(f"F1: {avg_metrics[1]:.4f}")
    print(f"Recall: {avg_metrics[2]:.4f}")
    print(f"Precision: {avg_metrics[3]:.4f}")
    print(f"Acc: {avg_metrics[4]:.4f}")
    print(f"F2: {avg_metrics[5]:.4f}")
    print(f"Mean IoU (Jaccard): {avg_iou:.4f}")
    print(f"Mean Dice: {avg_dice:.4f}")

    # 保存结果到txt
    result_txt = os.path.join(save_path, "result.txt")
    with open(result_txt, "w") as f:
        f.write(f"Jaccard (IoU): {avg_metrics[0]:.4f}\n")
        f.write(f"F1: {avg_metrics[1]:.4f}\n")
        f.write(f"Recall: {avg_metrics[2]:.4f}\n")
        f.write(f"Precision: {avg_metrics[3]:.4f}\n")
        f.write(f"Acc: {avg_metrics[4]:.4f}\n")
        f.write(f"F2: {avg_metrics[5]:.4f}\n")
        f.write(f"Mean IoU (Jaccard): {avg_iou:.4f}\n")
        f.write(f"Mean Dice: {avg_dice:.4f}\n")

    print(f"预测结果和指标已保存到：{save_path}")


