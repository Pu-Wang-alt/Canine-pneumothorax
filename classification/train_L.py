# ----------------- 导入必要的库 -----------------
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
import open_clip
import torch.nn.functional as F

# --- 混合精度训练所需的库 (保持不变) ---
from torch.cuda.amp import autocast, GradScaler

# ----------------- 从您的文件中导入 -----------------
from data_loader import DataTransforms, CroppedClassificationDataset, collate_fn


# ==================================================================
#               SECTION 1: VLM 模型定义
# ==================================================================
class VLMClassifier(nn.Module):
    """
    一个使用 OpenCLIP 图像编码器作为骨干的分类模型。
    增加了对骨干网络解冻微调的支持。
    """

    # <--- 修改 1: 增加 unfreeze_backbone 参数 --->
    def __init__(self, clip_model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', num_classes=2,
                 unfreeze_backbone=False):
        super().__init__()
        print(f"--- 步骤 3.1: 加载 OpenCLIP 模型 '{clip_model_name}' (预训练: '{pretrained}') ---")
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained)

        # <--- 修改 2: 根据 unfreeze_backbone 决定是否冻结参数 --->
        if not unfreeze_backbone:
            print("--- 模式: 冻结骨干网络，仅训练分类头 ---")
            for param in clip_model.parameters():
                param.requires_grad = False
        else:
            print("--- 模式: 解冻骨干网络，进行全模型微调 ---")
            # 所有参数默认 requires_grad=True，无需额外操作

        self.image_encoder = clip_model.visual
        feature_dim = self.image_encoder.output_dim

        self.classifier_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        print(f"--- 模型特征维度: {feature_dim}, 分类头已创建 ---")

    def forward(self, image):
        image_features = self.image_encoder(image)
        logits = self.classifier_head(image_features)
        return logits


# ==================================================================
#               SECTION 2: 损失/评估/验证函数 (保持您的版本)
# ==================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.full_like(targets, 1 - self.alpha, dtype=torch.float)
        alpha_t[targets == 1] = self.alpha
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


print("--- 使用 Focal Loss ---")
criterion = FocalLoss(alpha=0.85, gamma=2)


def validate(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for data in progress_bar:
            if data is None or data[0].size(0) == 0: continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
            pred_class = torch.argmax(outputs, dim=1)
            all_labels.append(labels.cpu())
            all_preds.append(pred_class.cpu())

    if not all_labels: return 0.0, 0.0, 0.0

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    TP = ((all_preds == 1) & (all_labels == 1)).sum().item()
    FP = ((all_preds == 1) & (all_labels == 0)).sum().item()
    FN = ((all_preds == 0) & (all_labels == 1)).sum().item()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"\nValidation -> Recall: {recall:.4f}, Precision: {precision:.4f}, F1-Score: {f1_score:.4f}")
    model.train()
    return f1_score, recall, precision


# ==================================================================
#               SECTION 3: 主训练流程
# ==================================================================
def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 使用设备: {device} ---")

    # <--- 修改 3: 建议为新模型创建一个新的保存目录 --->
    model_dir = os.path.join(os.getcwd(), 'saved_models_VLM_ViT-L-14_Finetune')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- 数据加载 (保持不变) ---
    print("--- 步骤 1: 创建数据转换器 ---")
    transforms_manager = DataTransforms(output_size=224)
    print("\n--- 步骤 2: 创建数据集和加载器 (使用智能裁剪) ---")
    train_dataset = CroppedClassificationDataset(csv_file=opt.train_csv, root_dir=opt.train_image_dir,
                                                 mask_dir=opt.train_mask_dir, transform=transforms_manager.train)
    val_dataset = CroppedClassificationDataset(csv_file=opt.val_csv, root_dir=opt.val_image_dir,
                                               mask_dir=opt.val_mask_dir, transform=transforms_manager.val)
    print(f"--- 找到 {len(train_dataset)} 张训练图片 ---")
    print(f"--- 找到 {len(val_dataset)} 张验证图片 ---")
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=8,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=8, collate_fn=collate_fn,
                            pin_memory=True)

    # --- 模型、优化器和混合精度设置 ---
    print("\n--- 步骤 3: 初始化VLM分类模型 (ViT-L-14) ---")
    # <--- 修改 4: 加载 ViT-L-14 模型并解冻 --->
    net = VLMClassifier(
        clip_model_name='ViT-L-14',
        pretrained='laion2b_s32b_b82k',
        num_classes=2,
        unfreeze_backbone=True  # <-- 关键！
    )
    net = net.to(device)

    # <--- 修改 5: 设置差分学习率 --->
    print("--- 设置差分学习率 ---")
    param_groups = [
        {'params': net.image_encoder.parameters(), 'lr': opt.backbone_lr},  # 骨干网络使用非常小的学习率
        {'params': net.classifier_head.parameters(), 'lr': opt.lr}  # 分类头使用较大的学习率
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)  # 使用 AdamW 优化器，更适合 Transformer
    scaler = GradScaler()
    print(f"--- VLM 模型和优化器已定义 (骨干 LR: {opt.backbone_lr}, 分类头 LR: {opt.lr}) ---")

    # --- 训练循环 ---
    best_f1_score = 0.0
    accumulation_steps = opt.accumulation_batch_size // opt.batchsize
    print(
        f"\n--- 梯度累积步数: {accumulation_steps} (实际批次: {opt.batchsize}, 等效批次: {opt.accumulation_batch_size}) ---")

    print("\n--- 开始微调训练 ---")
    for epoch in range(opt.epoch):
        net.train()
        running_loss = 0.0
        if len(train_loader) == 0:
            print(f"Epoch {epoch + 1}/{opt.epoch}: 训练数据加载器为空，跳过此epoch。")
            continue
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{opt.epoch}")
        optimizer.zero_grad()

        for i, data in progress_bar:
            if data is None: continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                # <--- 修改 6: 对所有参数进行梯度裁剪 --->
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix(loss=f'{running_loss / (i + 1):.4f}')

        val_f1, val_recall, val_precision = validate(net, val_loader, device)
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            save_path = os.path.join(model_dir, 'best_vlm_finetuned_model.pth')
            # <--- 修改 7: 保存整个模型的状态字典 --->
            torch.save(net.state_dict(), save_path)
            print(f"✅ [最佳模型] 已更新并保存至: {save_path} (F1-Score: {best_f1_score:.4f})")

        if (epoch + 1) % 5 == 0:  # <-- 可以调整保存频率
            periodic_save_path = os.path.join(model_dir, f'epoch_{epoch + 1}_vlm_finetuned.pth')
            # <--- 修改 8: 保存整个模型的状态字典 --->
            torch.save(net.state_dict(), periodic_save_path)
            print(f"💾 [定期保存] 模型已保存至: {periodic_save_path}")

    print("--- 训练完成 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- 路径参数 (保持您的默认值) ---
    parser.add_argument('--train_image_dir', type=str, default="/home/wangpu/wangpu/image_seg/dataset/Train/Images",
                        help='训练图片文件夹路径')
    parser.add_argument('--val_image_dir', type=str, default="/home/wangpu/wangpu/image_seg/dataset/Valid/Images",
                        help='验证图片文件夹路径')
    parser.add_argument('--train_mask_dir', type=str, default="/opt/predictions/Train",
                        help='训练集对应的mask文件夹路径')
    parser.add_argument('--val_mask_dir', type=str, default="/opt/predictions/Valid", help='验证集对应的mask文件夹路径')
    parser.add_argument('--train_csv', type=str, default="/home/wangpu/wangpu/classify/Resnet-50/cssv/Train.csv",
                        help='训练集CSV文件路径')
    parser.add_argument('--val_csv', type=str, default="/home/wangpu/wangpu/classify/Resnet-50/cssv/Valid.csv",
                        help='验证集CSV文件路径')

    # --- 训练超参数 (针对微调进行调整) ---
    # <--- 修改 9: 调整超参数默认值 --->
    parser.add_argument('--epoch', type=int, default=15, help='微调时的训练轮次')
    parser.add_argument('--lr', type=float, default=1e-4, help='分类头的学习率')
    parser.add_argument('--backbone_lr', type=float, default=2e-5, help='骨干网络的学习率 (应远小于分类头)')
    parser.add_argument('--batchsize', type=int, default=8, help='实际批处理大小 (ViT-L-14显存占用大, 从小开始尝试)')
    parser.add_argument('--accumulation_batch_size', type=int, default=32, help='梯度累积后的等效批处理大小')

    opt = parser.parse_args()
    print("=" * 20 + " 参数配置 " + "=" * 20)
    for k, v in vars(opt).items():
        print(f"{k}: {v}")
    print("=" * 50)
    main(opt)