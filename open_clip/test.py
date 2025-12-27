# evaluate.py

import torch
import argparse
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

# ==================================================================
#               SECTION 1: 关键依赖导入
# ==================================================================
# 从您现有的脚本中导入必要的类
try:
    from train_L import VLMClassifier
    from data_loader import DataTransforms, CroppedClassificationDataset, collate_fn
except ImportError as e:
    print(f"错误: 无法导入必要的类 ({e})。")
    print("请确保 evaluate.py, train_vlm_finetune.py, 和 data_loader.py 在同一个文件夹下。")
    exit()

# --- PyTorch & Sklearn ---
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# MODIFIED: 新增用于绘图和曲线计算的库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from matplotlib.colors import LogNorm

# 定义类别名称，必须与训练时一致
CLASS_NAMES = ["Normal", "Pneumothorax"]


# ==================================================================
#               SECTION 2: 新增的绘图与数据保存函数
# ==================================================================

def plot_confusion_matrix(cm, save_path):
    """绘制高度定制化的混淆矩阵热力图。"""
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=True,
                xticklabels=False, yticklabels=False, annot_kws={"size": 20},
                norm=LogNorm())
    # ax = plt.gca()
    # text_props = {'fontsize': 22, 'fontweight': 'bold', 'color': 'orange', 'transform': ax.transAxes}
    # ax.text(0.05, 0.95, 'TN', ha='left', va='top', **text_props)
    # ax.text(0.95, 0.95, 'FP', ha='right', va='top', **text_props)
    # ax.text(0.05, 0.05, 'FN', ha='left', va='bottom', **text_props)
    # ax.text(0.95, 0.05, 'TP', ha='right', va='bottom', **text_props)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 定制化混淆矩阵图已保存至: {save_path}")
    plt.close()


def save_curve_data(all_labels, all_probs, model_name, save_dir='curve_data'):
    """计算并保存用于后续组合绘图的ROC和P-R曲线数据。"""
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    curve_data = {
        'model_name': model_name,
        'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
        'precision': precision, 'recall': recall, 'pr_auc': pr_auc
    }
    save_path = os.path.join(save_dir, f'{model_name}_curve_data.npy')
    np.save(save_path, curve_data)
    print(f"✅ ROC/P-R 曲线数据已保存至: {save_path}")


# ==================================================================
#               SECTION 3: 核心评估函数 (有修改)
# ==================================================================

def load_model_for_evaluation(model_path, device):
    """加载微调好的 ViT-L/14 模型用于评估。(保持不变)"""
    print(f"--- 正在加载模型权重: {model_path} ---")
    model = VLMClassifier(clip_model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', num_classes=2,
                          unfreeze_backbone=True)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("--- 模型加载成功 ---")
    return model


# MODIFIED: 此函数现在额外返回概率值 (all_probs)
def run_evaluation(model, data_loader, device):
    """在给定的数据集上运行模型并收集所有预测、标签和概率。"""
    all_labels = []
    all_preds = []
    all_probs = []  # <--- 新增：用于存储正类的概率

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="正在评估数据集"):
            if batch is None: continue
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # --- 获取预测标签 ---
            preds = torch.argmax(outputs, dim=1)

            # --- 获取正类 (Pneumothorax, Class 1) 的概率 ---
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())  # <--- 收集概率

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(labels, preds):
    """计算并打印详细的评估指标。(保持不变)"""
    print("\n" + "=" * 60)
    print(" " * 20 + "评估结果报告")
    print("=" * 60)
    print("\n[混淆矩阵]")
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.columns.name = 'Predicted';
    cm_df.index.name = 'Actual'
    print(cm_df)
    print(
        f"\n- True Positives (TP): {tp}\n- True Negatives (TN): {tn}\n- False Positives (FP): {fp}\n- False Negatives (FN): {fn}")
    print("\n[分类指标详情]")
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    print(report)
    accuracy = accuracy_score(labels, preds)
    print(f"\n[总体准确率 (Accuracy)]\n{accuracy:.4f}")
    print("=" * 60)


# ==================================================================
#               SECTION 4: 主执行逻辑 (有修改)
# ==================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    print(f"--- 使用设备: {device} ---")

    model = load_model_for_evaluation(args.model_path, device)

    print("\n--- 准备评估数据集 ---")
    eval_transforms = DataTransforms(output_size=224).val
    eval_dataset = CroppedClassificationDataset(csv_file=args.csv_file, root_dir=args.image_dir, mask_dir=args.mask_dir,
                                                transform=eval_transforms)
    if len(eval_dataset) == 0:
        print(f"错误: 在 {args.csv_file} 中没有找到有效的数据。")
        return
    print(f"--- 找到 {len(eval_dataset)} 张评估图片 ---")
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                             collate_fn=collate_fn)

    # MODIFIED: 接收新增的概率值
    true_labels, predicted_labels, predicted_probs = run_evaluation(model, eval_loader, device)

    # 1. 打印文本格式的评估报告 (保持不变)
    print_metrics(true_labels, predicted_labels)

    # MODIFIED: 2. 生成可视化的图表和数据文件
    print("\n--- 正在生成可视化结果与数据文件 ---")
    cm = confusion_matrix(true_labels, predicted_labels)
    plot_confusion_matrix(cm, f'{args.model_name}_confusion_matrix.png')
    save_curve_data(true_labels, predicted_probs, args.model_name)
    print("--- 所有任务完成 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估微调后的 ViT-L/14 气胸检测模型")

    # MODIFIED: 新增 model_name 参数，用于文件命名
    parser.add_argument('--model_name', type=str, default='VLM_Finetuned',
                        help='为当前测试的模型指定一个名称，用于保存输出文件。')

    parser.add_argument('--model_path', type=str,
                        default="/home/wangpu/wangpu/classify/open_clip/saved_models_VLM_ViT-L-14_Finetune/best_vlm_finetuned_model.pth",
                        help='指向训练好的模型权重文件 (.pth) 的路径。')
    parser.add_argument('--image_dir', type=str, default="/home/wangpu/wangpu/image_seg/dataset/Test/Images",
                         help='评估图片所在的文件夹路径。')
    parser.add_argument('--csv_file', type=str, default="/home/wangpu/wangpu/classify/Resnet-50/cssv/Test.csv",
                        help='包含图片文件名和真实标签的CSV文件路径。')
    parser.add_argument('--mask_dir', type=str, default="/opt/predictions/Test", help='评估图片对应的mask文件夹路径。')
    parser.add_argument('--batch_size', type=int, default=16, help='评估时使用的批处理大小。')
    parser.add_argument('--use_cpu', action='store_true', help='强制使用CPU进行评估。')

    args = parser.parse_args()
    main(args)