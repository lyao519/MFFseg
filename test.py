import os
import random
import torch
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# 根据你的实际模型路径导入
from network.MedSegNetV2 import MedSegNetV2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_pred_from_output(output):
    """
    兼容模型输出为 Tensor / dict / list / tuple 的情况
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        # 常见键名优先
        for key in ["pred", "out", "mask", "seg", "main_out", "final"]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]

        # 找第一个 tensor
        for k, v in output.items():
            if torch.is_tensor(v):
                print(f"[Info] 使用字典中的输出键: {k}")
                return v

        raise TypeError(f"模型返回 dict，但其中没有 Tensor。keys={list(output.keys())}")

    if isinstance(output, (list, tuple)):
        for i, v in enumerate(output):
            if torch.is_tensor(v):
                print(f"[Info] 使用 list/tuple 中第 {i} 个输出")
                return v
        raise TypeError("模型返回 list/tuple，但其中没有 Tensor。")

    raise TypeError(f"无法处理的模型输出类型: {type(output)}")


def ensure_mask_shape(pred_mask):
    """
    统一成 [B, 1, H, W]
    """
    if pred_mask.dim() == 4:
        return pred_mask
    elif pred_mask.dim() == 3:
        return pred_mask.unsqueeze(1)
    elif pred_mask.dim() == 2:
        return pred_mask.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"不支持的 pred_mask 维度: {pred_mask.shape}")


def process_mask(y_pred):
    """
    输入: Tensor, shape [B, 1, H, W]
    输出: 可视化 RGB mask, shape [H, W, 3]
    """
    y_pred = y_pred[0].detach().cpu().numpy()  # [1, H, W]
    y_pred = np.squeeze(y_pred, axis=0)        # [H, W]
    y_pred = (y_pred > 0.5).astype(np.uint8) * 255
    y_pred = np.expand_dims(y_pred, axis=-1)   # [H, W, 1]
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred


def dice_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return (2 * np.sum(y_true * y_pred) + 1e-7) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)


def jaccard_score_np(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)


def load_image(img_path, size):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = torch.from_numpy(image).unsqueeze(0).float()
    return image


def load_mask(mask_path, size):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取 mask: {mask_path}")
    mask = cv2.resize(mask, size)
    mask = mask.astype(np.float32) / 255.0
    mask_bin = (mask > 0.5).astype(np.uint8)
    mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0).float()
    return mask_tensor


def evaluate(model, test_image_paths, test_mask_paths, save_dir, size=(256, 256), device="cuda"):
    model.eval()
    metrics_sum = np.zeros(6, dtype=np.float64)
    total = len(test_image_paths)

    if total == 0:
        raise ValueError("测试图片数量为 0，请检查测试集路径。")

    if len(test_image_paths) != len(test_mask_paths):
        raise ValueError(
            f"测试图片和 mask 数量不一致: images={len(test_image_paths)}, masks={len(test_mask_paths)}"
        )

    mask_save_dir = os.path.join(save_dir, "mask")
    create_dir(save_dir)
    create_dir(mask_save_dir)

    first_output_logged = False

    for img_path, mask_path in tqdm(zip(test_image_paths, test_mask_paths), total=total):
        # --- 加载与预处理 ---
        image = load_image(img_path, size).to(device)
        mask_tensor = load_mask(mask_path, size).to(device)

        # --- 推理 ---
        with torch.no_grad():
            output = model(image)

            if not first_output_logged:
                print(f"[Info] 模型输出类型: {type(output)}")
                if isinstance(output, dict):
                    print(f"[Info] 模型输出 keys: {list(output.keys())}")
                elif isinstance(output, (list, tuple)):
                    print(f"[Info] 模型输出长度: {len(output)}")
                first_output_logged = True

            pred_mask = get_pred_from_output(output)
            pred_mask = ensure_mask_shape(pred_mask)

            # 若模型输出通道数 > 1，默认取第 1 个通道
            if pred_mask.shape[1] > 1:
                print(f"[Warn] pred_mask 通道数为 {pred_mask.shape[1]}，默认取第 1 个通道")
                pred_mask = pred_mask[:, 0:1, :, :]

            # 一般分割网络输出 logits，做 sigmoid
            pred_mask = torch.sigmoid(pred_mask)

            pred_mask_bin = (pred_mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
            mask_bin_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)

        # --- 评价指标 ---
        dice = dice_score(mask_bin_np, pred_mask_bin)
        jaccard = jaccard_score_np(mask_bin_np, pred_mask_bin)
        acc = accuracy_score(mask_bin_np.flatten(), pred_mask_bin.flatten())

        intersection = np.sum(mask_bin_np * pred_mask_bin)
        recall = intersection / (np.sum(mask_bin_np) + 1e-7)
        precision = intersection / (np.sum(pred_mask_bin) + 1e-7)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-7)

        metrics = np.array([jaccard, dice, recall, precision, acc, f2], dtype=np.float64)
        metrics_sum += metrics

        # --- 保存预测结果 ---
        img_name = os.path.splitext(os.path.basename(mask_path))[0]
        pred_vis = process_mask(pred_mask)
        cv2.imwrite(os.path.join(mask_save_dir, f"{img_name}.png"), pred_vis)

    # --- 平均指标 ---
    avg_metrics = metrics_sum / total

    print("\n==== 平均评价指标 ====")
    print(f"Jaccard (IoU): {avg_metrics[0]:.4f}")
    print(f"Dice:          {avg_metrics[1]:.4f}")
    print(f"Recall:        {avg_metrics[2]:.4f}")
    print(f"Precision:     {avg_metrics[3]:.4f}")
    print(f"Accuracy:      {avg_metrics[4]:.4f}")
    print(f"F2:            {avg_metrics[5]:.4f}")

    with open(os.path.join(save_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"Jaccard (IoU): {avg_metrics[0]:.4f}\n")
        f.write(f"Dice:          {avg_metrics[1]:.4f}\n")
        f.write(f"Recall:        {avg_metrics[2]:.4f}\n")
        f.write(f"Precision:     {avg_metrics[3]:.4f}\n")
        f.write(f"Accuracy:      {avg_metrics[4]:.4f}\n")
        f.write(f"F2:            {avg_metrics[5]:.4f}\n")

    print(f"\n保存预测 mask 和指标至: {save_dir}")


if __name__ == "__main__":
    # ========== 配置 ==========
    set_seed(42)

    dataset = "ISIC2018"
    size = (256, 256)
    save_dir = f"results/{dataset}/tversky_MedSegNetV2"
    checkpoint_path = r""
    test_root = f"./data/{dataset}/"
    test_img_dir = os.path.join(test_root, "Test_Images")
    test_mask_dir = os.path.join(test_root, "Test_Masks")

    # ========== 路径读取 ==========
    test_imgs = sorted([
        os.path.join(test_img_dir, f)
        for f in os.listdir(test_img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])
    test_masks = sorted([
        os.path.join(test_mask_dir, f)
        for f in os.listdir(test_mask_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])

    print(f"测试图像数量: {len(test_imgs)}")
    print(f"测试掩码数量: {len(test_masks)}")

    # ========== 模型准备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = MedSegNetV2()

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        print("[Info] 检测到 checkpoint 中包含 model_state_dict，按该字段加载")
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[Info] 直接加载 checkpoint 作为 state_dict")
        model.load_state_dict(ckpt)

    model.to(device)

    # ========== 评测 ==========
    evaluate(model, test_imgs, test_masks, save_dir, size=size, device=device)
