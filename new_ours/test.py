# ----------------- 导入必要的库 -----------------
import os
import torch
import numpy as np
import glob
import argparse
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from torchvision import transforms
import open_clip  # VLM需要用它来编码文本

# ----------------- 从您的项目中导入 -----------------
from data_loader import SalObjDataset, RescaleT, ToTensor
from attentional_vlm_model import FullSegmentationModel  # 导入您的VLM模型


# ==================================================================
#               SECTION 1: 辅助函数
# ==================================================================

def compute_metrics(pred, gt, eps=1e-6):
    """计算 mDice 和 mIoU"""
    # 确保输入是布尔类型或0/1整数
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    intersection = np.logical_and(pred_bin, gt_bin)
    union = np.logical_or(pred_bin, gt_bin)

    miou = (np.sum(intersection) + eps) / (np.sum(union) + eps)
    mdice = (2. * np.sum(intersection) + eps) / (np.sum(pred_bin) + np.sum(gt_bin) + eps)

    return miou, mdice


# ==================================================================
#               SECTION 2: 主评估函数 (适配 VLM)
# ==================================================================
def evaluate(model, test_loader, device, text_prompt):
    """使用 VLM 模型和文本提示进行评估"""
    model.eval()
    total_miou = 0.0
    total_mdice = 0.0

    # 为VLM模型准备文本输入
    print(f"--- 使用评估提示: '{text_prompt}' ---")
    text_tokens = open_clip.tokenize([text_prompt]).to(device)

    progress_bar = tqdm(test_loader, desc="计算指标中", leave=False)

    with torch.no_grad():
        for data in progress_bar:
            # 数据加载器已经完成了图像预处理
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels_np = labels.squeeze(1).numpy()  # GT转为Numpy用于后续计算

            # 扩展文本token以匹配批次大小
            batch_size = inputs.shape[0]
            text_tokens_batch = text_tokens.expand(batch_size, -1)

            # 使用 VLM 模型进行预测
            outputs = model(inputs, text_tokens_batch)

            # 后处理模型输出
            # squeeze(1) 移除通道维度 -> sigmoid转为概率 -> >0.5二值化
            preds_mask = (torch.sigmoid(outputs.squeeze(1)) > 0.5)
            preds_mask_np = preds_mask.cpu().numpy()

            # 逐个样本计算指标
            for i in range(batch_size):
                miou, mdice = compute_metrics(preds_mask_np[i], labels_np[i])
                total_miou += miou
                total_mdice += mdice

            progress_bar.set_postfix(mDice=f'{total_mdice / ((progress_bar.n + 1) * batch_size):.4f}',
                                     mIoU=f'{total_miou / ((progress_bar.n + 1) * batch_size):.4f}')

    num_samples = len(test_loader.dataset)
    avg_miou = total_miou / num_samples
    avg_mdice = total_mdice / num_samples

    print("\n" + "=" * 30)
    print("      VLM 评估结果")
    print("=" * 30)
    print(f"  测试样本数: {num_samples}")
    print(f"  平均 mIoU: {avg_miou:.4f}")
    print(f"  平均 mDice: {avg_mdice:.4f}")
    print("=" * 30)


# ==================================================================
#               SECTION 3: 可视化函数 (适配 VLM)
# ==================================================================
def visualize_predictions(model, path, device, text_prompt, transform, num_images=10):
    """随机抽取图片，使用 VLM 进行预测并保存可视化结果"""
    print("--- 开始生成 VLM 预测效果图 ---")

    save_dir = os.path.join(os.getcwd(), "test_results_VLM")
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果将保存在 '{save_dir}' 文件夹中...")

    # 准备文本输入
    text_tokens = open_clip.tokenize([text_prompt]).to(device)

    image_root = os.path.join(path, 'Images/')
    gt_root = os.path.join(path, 'Masks/')

    image_files = sorted(glob.glob(os.path.join(image_root, '*.jpg')))
    selected_files = random.sample(image_files, min(num_images, len(image_files)))

    model.eval()
    with torch.no_grad():
        for img_path in tqdm(selected_files, desc="生成可视化图片"):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(gt_root, base_name + '.png')

            if not os.path.exists(gt_path):
                continue

            # 加载原始 PIL 图像和GT用于显示
            original_image = Image.open(img_path).convert('RGB')
            gt_mask = Image.open(gt_path).convert('L')

            # 将 PIL 图像转换为 NumPy 数组
            original_image_np = np.array(original_image)
            gt_mask_np = np.array(gt_mask)

            # 确保 mask 是三维的 (H, W, 1)
            if len(gt_mask_np.shape) == 2:
                gt_mask_np = gt_mask_np[:, :, np.newaxis]

            # 创建完整的样本字典
            sample_dict = {
                'image': original_image_np,
                'label': gt_mask_np,
                'imidx': np.array([0]),  # 使用numpy数组
                'class_label': 0  # 默认值，因为测试时可能没有CSV文件
            }

            # 应用transform
            transformed_data = transform(sample_dict)
            inputs = transformed_data['image'].unsqueeze(0).to(device)

            # 使用 VLM 模型进行预测
            outputs = model(inputs, text_tokens)
            pred_mask = (torch.sigmoid(outputs.squeeze(0).squeeze(0)) > 0.5).cpu().numpy()

            # 开始绘图
            plt.figure(figsize=(18, 6))
            plt.suptitle(f"Prompt: {text_prompt}", fontsize=16)

            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(original_image)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('Ground Truth Mask')
            plt.imshow(gt_mask, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('VLM Predicted Mask')
            plt.imshow(pred_mask, cmap='gray')
            plt.axis('off')

            save_path = os.path.join(save_dir, os.path.basename(img_path))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    print(f"--- 可视化图片已全部保存在 '{save_dir}' ---")


# ==================================================================
#               SECTION 4: 主函数 (适配 VLM)
# ==================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str,
                        default='/home/wangpu/wangpu/heart/data10/test_images',
                        help='path to test dataset')

    # VLM 的参数
    parser.add_argument('--model_path', type=str,
                        default='/home/wangpu/wangpu/image_seg/new_ours/best_model.pth',
                        help='path to the trained VLM checkpoint file (.pth)')
    parser.add_argument('--train_size', type=int, default=224,
                        help='input image size, must match training size')
    parser.add_argument('--prompt', type=str,
                        default='a collapsed lung due to pneumothorax',
                        help='The text prompt to guide the segmentation model')

    # 新增：CSV文件路径参数（可选）
    parser.add_argument('--csv_path', type=str, default=None,
                        help='path to CSV file containing classification labels (optional)')

    args = parser.parse_args()

    # 1. 设备和模型加载
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 使用设备: {device} ---")

    print(f"--- 加载 VLM 模型: {args.model_path} ---")
    # 实例化模型
    net = FullSegmentationModel(
        clip_model_name='ViT-L-14',
        pretrained='laion2b_s32b_b82k',
        unfreeze_backbone=True
    )
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)
    net.eval()

    # 2. 数据加载 - 修复：添加csv_path参数
    test_transform = transforms.Compose([
        RescaleT(args.train_size),
        ToTensor()
    ])

    test_img_name_list = glob.glob(os.path.join(args.test_path, 'Images', '*.jpg'))
    test_lbl_name_list = [os.path.join(args.test_path, 'Masks',
                                       os.path.splitext(os.path.basename(p))[0] + '.png')
                          for p in test_img_name_list]

    # 修复：创建数据集时传入csv_path参数
    test_dataset = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        transform=test_transform,
        csv_path=args.csv_path  # 传入CSV路径（可以是None）
    )

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    print(f"--- 找到 {len(test_dataset)} 张测试图片 ---")

    # 3. 执行评估和可视化
    evaluate(net, test_loader, device, args.prompt)
    visualize_predictions(net, args.test_path, device, args.prompt,
                          transform=test_transform, num_images=10)