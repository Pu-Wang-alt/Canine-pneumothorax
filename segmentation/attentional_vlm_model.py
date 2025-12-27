# attentional_vlm_model.py (最终修正版 - 解决通道数不匹配)
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class UNetDecoder(nn.Module):
    # 这个Decoder的结构现在是正确的，因为我们将会在VLM_UNet中确保输入给它的通道数是正确的
    def __init__(self, out_channels=1):
        super(UNetDecoder, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 这是标准U-Net的维度设计，我们将坚持这个设计
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)  # cat(512, 512) -> 1024
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)  # cat(256, 256) -> 512
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)  # cat(128, 128) -> 256
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)  # cat(64, 64) -> 128
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, enc_features, bottleneck):
        enc1, enc2, enc3, enc4 = enc_features['enc1'], enc_features['enc2'], enc_features['enc3'], enc_features['enc4']
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # 现在输入维度会是 512+512=1024
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # 256+256=512
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # 128+128=256
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # 64+64=128
        dec1 = self.decoder1(dec1)
        return self.final_conv(dec1)


class VLM_UNet(nn.Module):
    def __init__(self, clip_model, out_channels=1, unfreeze_backbone=False):
        super(VLM_UNet, self).__init__()
        self.clip_model = clip_model
        if not unfreeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        vision_width = 1024
        text_width = 768

        # <--- 最终修正: 增加投影层来“压缩”Skip-Connection的通道维度 --->
        self.skip_proj_4 = nn.Conv2d(vision_width, 512, kernel_size=1)
        self.skip_proj_3 = nn.Conv2d(vision_width, 256, kernel_size=1)
        self.skip_proj_2 = nn.Conv2d(vision_width, 128, kernel_size=1)
        self.skip_proj_1 = nn.Conv2d(vision_width, 64, kernel_size=1)

        self.text_projection = nn.Linear(text_width, vision_width)
        self.visual_encoder = self.clip_model.visual
        self.decoder = UNetDecoder(out_channels)

    def forward(self, image, text_tokens, return_features=False):
        x = self.visual_encoder.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                     dtype=x.dtype, device=x.device),
                       x], dim=1)
        x = x + self.visual_encoder.positional_embedding.to(x.dtype)
        x = self.visual_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual_encoder.transformer(x)
        x = x.permute(1, 0, 2)
        image_features_seq = x

        text_features = self.clip_model.encode_text(text_tokens)

        image_grid_features = image_features_seq[:, 1:, :]
        bs, _, width = image_grid_features.shape
        grid_size = int((image_grid_features.shape[1]) ** 0.5)
        bottleneck = image_grid_features.permute(0, 2, 1).view(bs, width, grid_size, grid_size)

        # <--- 最终修正: 使用投影层处理Skip-Connection --->
        enc4_unproj = F.interpolate(bottleneck, size=(bottleneck.shape[2] * 2, bottleneck.shape[3] * 2),
                                    mode='bilinear', align_corners=False)
        enc3_unproj = F.interpolate(bottleneck, size=(bottleneck.shape[2] * 4, bottleneck.shape[3] * 4),
                                    mode='bilinear', align_corners=False)
        enc2_unproj = F.interpolate(bottleneck, size=(bottleneck.shape[2] * 8, bottleneck.shape[3] * 8),
                                    mode='bilinear', align_corners=False)
        enc1_unproj = F.interpolate(bottleneck, size=(bottleneck.shape[2] * 16, bottleneck.shape[3] * 16),
                                    mode='bilinear', align_corners=False)

        enc_features = {
            'enc4': self.skip_proj_4(enc4_unproj),  # 1024 -> 512 channels
            'enc3': self.skip_proj_3(enc3_unproj),  # 1024 -> 256 channels
            'enc2': self.skip_proj_2(enc2_unproj),  # 1024 -> 128 channels
            'enc1': self.skip_proj_1(enc1_unproj),  # 1024 -> 64 channels
        }

        text_proj_vec = self.text_projection(text_features.float())
        text_proj_map = text_proj_vec.unsqueeze(-1).unsqueeze(-1)
        bottleneck_fused = bottleneck * torch.sigmoid(text_proj_map)

        mask_logits = self.decoder(enc_features, bottleneck_fused)
        initial_mask = F.interpolate(mask_logits, size=image.shape[2:], mode='bilinear', align_corners=False)

        if return_features:
            return initial_mask, bottleneck, text_proj_vec
        return initial_mask


# FullSegmentationModel 和 AttentionalVLM_FlowMatching 保持不变
# ... (将这两个类从之前的代码复制过来)
class AttentionalVLM_FlowMatching(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8, hidden_channels=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_encoder = nn.Sequential(nn.Conv2d(1, hidden_channels // 2, kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(hidden_channels // 2, embed_dim, kernel_size=1))
        self.positional_encoding = nn.Parameter(torch.randn(1, embed_dim, 32, 32))
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.v_predictor_ffn = nn.Sequential(nn.Conv2d(embed_dim, hidden_channels, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, kernel_size=1))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x_initial, image_features, text_features, steps=10):
        x = x_initial
        B, D, h, w = image_features.shape
        text_token = text_features.unsqueeze(1)
        image_seq = image_features.flatten(2).permute(0, 2, 1)
        kv = torch.cat([text_token, image_seq], dim=1)
        dt = 1.0 / steps
        for _ in range(steps):
            mask_feat_map = self.mask_encoder(torch.sigmoid(x))
            pos_encoding = F.interpolate(self.positional_encoding, size=mask_feat_map.shape[2:], mode='bilinear')
            query_map = mask_feat_map + pos_encoding
            query_seq = query_map.flatten(2).permute(0, 2, 1)
            query_norm = self.norm1(query_seq)
            attn_output, _ = self.cross_attention(query=query_norm, key=kv, value=kv)
            query_seq = query_seq + attn_output
            query_seq = self.norm2(query_seq)
            refined_map = query_seq.permute(0, 2, 1).view(B, D, x.shape[2], x.shape[3])
            v = self.v_predictor_ffn(refined_map)
            x = x + dt * v
        return x


class FullSegmentationModel(nn.Module):
    def __init__(self, clip_model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', unfreeze_backbone=False,
                 flow_steps=10):
        super().__init__()
        print("--- 正在构建完整的语义引导分割模型 ---")
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained)
        self.vlm_unet = VLM_UNet(clip_model, unfreeze_backbone=unfreeze_backbone)
        embed_dim = 1024
        self.attentional_flow = AttentionalVLM_FlowMatching(embed_dim=embed_dim)
        self.flow_steps = flow_steps

    def forward(self, image, text_tokens):
        initial_mask_logits, image_features, text_features = self.vlm_unet(image, text_tokens, return_features=True)
        final_mask_logits = self.attentional_flow(
            x_initial=initial_mask_logits,
            image_features=image_features,
            text_features=text_features,
            steps=self.flow_steps
        )
        return final_mask_logits