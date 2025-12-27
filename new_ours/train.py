# ----------------- 导入必要的库 -----------------
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- 新增: 导入F，因为structure_loss需要
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
import glob
import argparse
from tqdm import tqdm
import open_clip  # <--- 新增: 导入open_clip用于文本处理
import random  # <--- 新增: 用于随机选择prompt

# --- 混合精度训练所需的库 (保持不变) ---
from torch.cuda.amp import autocast, GradScaler

# ----------------- 从您的文件中导入 -----------------
from data_loader import RescaleT, RandomCrop, ToTensor, SalObjDataset
# <--- 修改: 导入新的VLM分割模型 --->
# from attentional_vlm_model import FullSegmentationModel
from attentional_vlm_model import FullSegmentationModel

# ==================================================================
#               SECTION 1: 损失函数和评估指标
# ==================================================================

# <--- 修改: 换回更适合分割任务的 structure_loss --->
# 这个函数直接从您最初的训练代码中复制而来
def structure_loss(pred, mask):
    """
    修改后的 structure_loss，能自动处理3D或4D输入。
    """
    # 确保 pred 和 mask 都是 4D 的 [B, 1, H, W]
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    # --- 后续计算逻辑完全保持不变 ---
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    # 这里的 wbce 和 weit 都是 4D，所以 .sum(dim=(2,3)) 可以正常工作
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

# <--- 修改: 更新 Dice 系数计算函数以适配新模型输出 --->
# preds_logits 是模型的单通道输出, shape: [B, H, W]
# gt 是真实的标签, shape: [B, H, W], type: float
def compute_metrics_torch(preds_logits, gt, eps=1e-6):
    """
    在 PyTorch 中计算 mDice 和 mIoU，输入为张量。

    Args:
        preds_logits (torch.Tensor): 模型的原始输出 (logits), shape: [B, H, W]。
        gt (torch.Tensor): 真实标签, shape: [B, H, W]。
        eps (float): 防止除以零的小常数。

    Returns:
        tuple: (iou_score, dice_score) 两个标量值。
    """
    # 1. 从 logits 得到二值化预测掩码
    pred_bin = (torch.sigmoid(preds_logits) > 0.5).float()
    gt_bin = (gt > 0.5).float()  # 确保 ground truth 也是二值化的

    # 2. 在空间维度 (H, W) 上计算交集和各自的和
    # 结果的 shape 为 [B]
    intersection = (pred_bin * gt_bin).sum(dim=(1, 2))
    pred_sum = pred_bin.sum(dim=(1, 2))
    gt_sum = gt_bin.sum(dim=(1, 2))

    # 3. 计算每个样本的 Dice 和 IoU
    # 结果的 shape 仍然是 [B]
    dice = (2. * intersection + eps) / (pred_sum + gt_sum + eps)
    union = pred_sum + gt_sum - intersection
    iou = (intersection + eps) / (union + eps)

    # 4. 返回整个批次的平均值
    return iou.mean().item(), dice.mean().item()


# ==================================================================
#               SECTION 2: 验证函数
# ==================================================================
# <--- 修改: 验证函数需要接收文本prompt --->
def validate(model, val_loader, device, text_prompts):
    """
    修改后的验证函数，使用 PyTorch 计算 mDice 和 mIoU，并返回 avg_dice。
    """
    model.eval()
    total_iou = 0.0
    total_dice = 0.0

    val_prompt = text_prompts['positive'][0]
    text_tokens = open_clip.tokenize([val_prompt]).to(device)

    progress_bar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for data in progress_bar:
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            batch_size = inputs.shape[0]
            text_tokens_batch = text_tokens.expand(batch_size, -1)

            # 使用 autocast 保持与训练逻辑一致
            with autocast():
                outputs = model(inputs, text_tokens_batch)

            # 使用新的 PyTorch 指标函数
            # outputs shape: [B, 1, H, W], labels shape: [B, 1, H, W]
            # 我们需要移除通道维度，使其变为 [B, H, W]
            batch_iou, batch_dice = compute_metrics_torch(outputs.squeeze(1), labels.squeeze(1))

            # 累加每个批次的平均分数
            total_iou += batch_iou
            total_dice += batch_dice

            # 更新进度条，显示运行时的平均指标
            running_avg_iou = total_iou / (progress_bar.n + 1)
            running_avg_dice = total_dice / (progress_bar.n + 1)
            progress_bar.set_postfix(mDice=f'{running_avg_dice:.4f}', mIoU=f'{running_avg_iou:.4f}')

    # 计算最终的平均指标
    avg_iou = total_iou / len(val_loader)
    avg_dice = total_dice / len(val_loader)

    # 打印更详细的验证结果
    print(f"\nValidation -> Avg Dice: {avg_dice:.4f}, Avg IoU: {avg_iou:.4f} (Prompt: '{val_prompt}')")

    # 将模型恢复到训练模式
    model.train()

    # 关键：只返回 dice 分数以兼容主训练循环中的模型保存逻辑
    return avg_dice


# ==================================================================
#               SECTION 3: 主训练流程
# ==================================================================
def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 使用设备: {device} ---")

    model_dir = os.path.join(os.getcwd(), 'saved_models_VLM_Flow')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 定义文本提示 (代码结构清晰，保持不变)
    text_prompts = {
        'positive': [
            "a collapsed lung due to pneumothorax",
            "pneumothorax in a canine chest x-ray",
            "the area of lung collapse",
            "free air in the chest cavity causing lung collapse",
            "radiograph showing a collapsed lung in a dog",
            "evidence of pneumothorax",
        ],
        'negative': [
            "healthy canine lungs",
            "a normal chest x-ray with no pneumothorax",
            "fully expanded lungs",
            "no signs of lung collapse",
            "a clear radiograph of a dog's chest",
            "no abnormalities detected in the lungs",
        ]
    }

    # --- 数据加载 (保持不变) ---
    train_img_name_list = glob.glob(os.path.join(opt.train_path, 'Images', '*.jpg'))
    train_lbl_name_list = [os.path.join(opt.train_path, 'Masks', os.path.splitext(os.path.basename(p))[0] + '.png') for
                           p in train_img_name_list]
    print(f"--- 找到 {len(train_img_name_list)} 张训练图片 ---")
    train_dataset = SalObjDataset(
        img_name_list=train_img_name_list, lbl_name_list=train_lbl_name_list,
        transform=transforms.Compose([RescaleT(opt.train_size), ToTensor()]),
        csv_path=opt.train_csv
    )
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True, num_workers=4)

    val_img_name_list = glob.glob(os.path.join(opt.val_path, 'Images', '*.jpg'))
    val_lbl_name_list = [os.path.join(opt.val_path, 'Masks', os.path.splitext(os.path.basename(p))[0] + '.png') for p in
                         val_img_name_list]
    print(f"--- 找到 {len(val_lbl_name_list)} 张验证图片 ---")
    val_dataset = SalObjDataset(
        img_name_list=val_img_name_list, lbl_name_list=val_lbl_name_list,
        transform=transforms.Compose([RescaleT(opt.train_size), ToTensor()]),
        csv_path=opt.val_csv
    )
    val_loader = DataLoader(val_dataset, batch_size=opt.batchsize, shuffle=False, num_workers=4)

    # --- 模型、优化器和混合精度设置 (保持不变) ---
    net = FullSegmentationModel(
        clip_model_name='ViT-L-14',
        pretrained='laion2b_s32b_b82k',
        unfreeze_backbone=True
    ).to(device)

    param_groups = [
        {'params': [p for p in net.vlm_unet.clip_model.parameters() if p.requires_grad], 'lr': opt.backbone_lr},
        {'params': net.vlm_unet.text_projection.parameters(), 'lr': opt.lr},
        {'params': net.vlm_unet.decoder.parameters(), 'lr': opt.lr},
        {'params': net.attentional_flow.parameters(), 'lr': opt.lr},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    scaler = GradScaler()
    print(f"--- VLM 模型和差分学习率优化器已定义 (主干 LR: {opt.backbone_lr}, Flow模块 LR: {opt.lr}) ---")

    if opt.load_weights is not None:
        if os.path.exists(opt.load_weights):
            print(f"--- 正在从权重文件加载模型: {opt.load_weights} ---")
            # 使用 map_location=device 确保权重能正确加载到当前设备
            try:
                net.load_state_dict(torch.load(opt.load_weights, map_location=device))
                print("✅ [加载成功] 模型权重已加载。将开始一次新的训练。")
            except Exception as e:
                print(f"❌ [加载失败] 加载权重时出错: {e}")
                # 如果加载失败，可以选择退出或继续从头训练
                return
        else:
            print(f"--- 警告: 找不到指定的权重文件: {opt.load_weights}。将从头开始训练。---")

    # <--- 新增: 断点续训加载逻辑 --->
    start_epoch = 0
    best_dice = 0.0
    checkpoint_path = os.path.join(model_dir, 'latest_checkpoint.pth')

    if opt.resume and os.path.exists(checkpoint_path):
        print(f"--- 正在从断点恢复: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_dice = checkpoint['best_dice']

        print(f"✅ [断点恢复] 模型、优化器和 Dice 分数已加载。将从 Epoch {start_epoch + 1} 开始。")
    else:
        print("--- 从头开始进行新的训练 ---")

    # --- 训练循环 ---
    accumulation_steps = max(1, opt.accumulation_batch_size // opt.batchsize)

    print("--- 开始VLM微调训练 ---")
    # <--- 修改: 调整 for 循环的起始点 --->
    for epoch in range(start_epoch, opt.epoch):
        net.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{opt.epoch}")
        optimizer.zero_grad()

        for i, data in progress_bar:
            inputs, labels, class_labels = data['image'], data['label'], data['class_label']
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            batch_prompts = [random.choice(text_prompts['positive' if lbl == 1 else 'negative']) for lbl in
                             class_labels]
            text_tokens = open_clip.tokenize(batch_prompts).to(device)

            with autocast():
                outputs = net(inputs, text_tokens)
                loss = structure_loss(outputs.squeeze(1), labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix(loss=f'{running_loss / (i + 1):.4f}')

        # --- Epoch结束后的验证和模型保存 ---
        val_dice = validate(net, val_loader, device, text_prompts)

        # 保存最佳模型逻辑保持不变，但依赖于从断点加载的 best_dice
        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(model_dir, 'best_model.pth')
            # 只保存模型权重，保持与原始逻辑一致
            torch.save(net.state_dict(), save_path)
            print(f"✅ [最佳模型] 已更新并保存至: {save_path} (Dice: {best_dice:.4f})")

        # 定期保存逻辑保持不变
        if (epoch + 1) % 10 == 0:
            periodic_save_path = os.path.join(model_dir, f'epoch_{epoch + 1}.pth')
            torch.save(net.state_dict(), periodic_save_path)
            print(f"💾 [定期保存] 模型已保存至: {periodic_save_path}")

        # <--- 新增: 在每个epoch结束后保存断点信息 --->
        # 注意: 我们保存的是下一个要开始的 epoch 编号
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_dice': best_dice,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"🔄 [断点保存] 检查点已保存至: {checkpoint_path}")

    print("--- 训练完成 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument('--train_path', type=str, default='/root/autodl-tmp/dataset/Train',
                        help='path to train dataset')
    parser.add_argument('--val_path', type=str, default='/root/autodl-tmp/dataset/Valid',
                        help='path to validation dataset')
    parser.add_argument('--train_csv', type=str, default='/home/cssv/Train.csv', help='path to train CSV file')
    parser.add_argument('--val_csv', type=str, default='/home/cssv/Valid.csv', help='path to validation CSV file')

    # 训练超参数
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for newly added modules (FlowMatching)')
    parser.add_argument('--backbone_lr', type=float, default=1e-5, help='learning rate for VLM backbone')
    parser.add_argument('--batchsize', type=int, default=2, help='training batch size (VLM-L is large, start small)')
    parser.add_argument('--accumulation_batch_size', type=int, default=16, help='effective batch size')
    parser.add_argument('--train_size', type=int, default=224,
                        help='training dataset size (must match CLIP model input size)')

    # <--- 新增: 控制是否从断点恢复的参数 --->
    parser.add_argument('--resume', action='store_true', help='resume training from latest checkpoint')
    parser.add_argument('--load_weights', type=str, default="",
                        help='仅加载模型权重文件 (.pth) 以开始新的训练')
    opt = parser.parse_args()

    print("=" * 20 + " 参数配置 " + "=" * 20)
    for k, v in vars(opt).items():
        print(f"{k}: {v}")
    print("=" * 50)

    main(opt)