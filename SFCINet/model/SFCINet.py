# -*- coding: utf-8 -*-
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 双分支均采用 MobileViT
from mobilevit import mobile_vit_small

# -----------------------------------------------------------
# 1. 基础算子 (垂直展开，毫无缩水)
# -----------------------------------------------------------
class CA(nn.Module):
    """标准的 Channel Attention"""
    def __init__(self, in_ch, reduction=16):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        mip = max(1, in_ch // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, mip, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mip, in_ch, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.fc(self.avg_pool(x))
        return x * weight

class SA(nn.Module):
    """标准的 Spatial Attention"""
    def __init__(self):
        super(SA, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        weight = torch.sigmoid(self.conv(x_cat))
        return weight

class DWBNReLU(nn.Module):
    """深度可分离卷积 + BN + ReLU"""
    def __init__(self, in_ch, out_ch):
        super(DWBNReLU, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): 
        return self.net(x)

# -----------------------------------------------------------
# 2. SFS: 空频协同模块 (Spatial-Frequency Synergy)
# -----------------------------------------------------------
class SFS(nn.Module):
    def __init__(self, channel):
        super(SFS, self).__init__()
        self.ca = CA(channel * 3)
        self.compress = nn.Conv2d(channel * 3, channel, kernel_size=1, bias=False)
        
        self.amp_proc = nn.Sequential(
            DWBNReLU(channel, channel), 
            nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        )
        self.phase_proc = nn.Sequential(
            DWBNReLU(channel, channel), 
            nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        )
        
        self.gate_gen = nn.Sequential(
            DWBNReLU(channel, channel * 2), 
            nn.Sigmoid()
        )
        self.r_refine = DWBNReLU(channel, channel)
        self.d_refine = DWBNReLU(channel, channel)

    def forward(self, r, d):
        f_cat = torch.cat([r + d, r, d], dim=1)
        f_ca = self.ca(f_cat)
        f_att = self.compress(f_ca)
        
        # 强制单精度执行 FFT
        freq = torch.fft.fft2(f_att.float(), norm='ortho')
        amp = torch.abs(freq).type_as(f_att)
        phase = torch.angle(freq).type_as(f_att)
        
        amp_enh = torch.abs(self.amp_proc(amp))
        phase_enh = self.phase_proc(phase)
        
        amp_enh_fp32 = amp_enh.float()
        phase_enh_fp32 = phase_enh.float()
        real = amp_enh_fp32 * torch.cos(phase_enh_fp32)
        imag = amp_enh_fp32 * torch.sin(phase_enh_fp32)
        
        s_enh = torch.fft.ifft2(torch.complex(real, imag), norm='ortho').real.type_as(f_att)
        
        gates = self.gate_gen(s_enh)
        m_r, m_d = torch.chunk(gates, 2, dim=1)
        
        r_f = self.r_refine(r)
        d_f = self.d_refine(d)
        
        r_out = r_f + (r_f * m_r)
        d_out = d_f + (d_f * m_d)
        
        return r_out, d_out, s_enh

# -----------------------------------------------------------
# 3. CMGI (跨模态引导交互) & FCA (特征校准)
# -----------------------------------------------------------
class CMGI(nn.Module):
    def __init__(self, channel):
        super(CMGI, self).__init__()
        self.r_refine = DWBNReLU(channel, channel)
        self.d_refine = DWBNReLU(channel, channel)
        self.sa_r = SA()
        self.sa_d = SA()
        self.fuse_conv = nn.Conv2d(channel * 2, channel, kernel_size=1, bias=False)
        self.fuse_refine = DWBNReLU(channel, channel)

    def forward(self, r, d):
        r_f = self.r_refine(r)
        d_f = self.d_refine(d)
        
        sa_r_weight = self.sa_r(r_f)
        sa_d_weight = self.sa_d(d_f)
        
        m_r = r_f * sa_d_weight
        m_d = d_f * sa_r_weight
        
        f_cat = torch.cat([m_r, m_d], dim=1)
        f_fused = self.fuse_refine(self.fuse_conv(f_cat))
        
        return f_fused + r_f + d_f

class FeatureCalibration(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FeatureCalibration, self).__init__()
        self.confidence_gate = CA(in_ch)
        self.refine_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x_weighted = self.confidence_gate(x)
        out = self.refine_conv(x_weighted)
        return out

class CHD(nn.Module):
    def __init__(self, channel):
        super(CHD, self).__init__()
        self.cmgi = CMGI(channel)
        self.fca = FeatureCalibration(channel * 2, channel)

    def forward(self, r, d, s_enh, p_prev=None):
        f_sa_fuse = self.cmgi(r, d)
        f_agg = torch.cat([f_sa_fuse, s_enh], dim=1)
        f_calibrated = self.fca(f_agg)
        
        if p_prev is not None:
            p_prev_up = F.interpolate(p_prev, size=f_calibrated.shape[2:], mode='bilinear', align_corners=False)
            out = f_calibrated + p_prev_up
        else:
            out = f_calibrated
            
        return out

# -----------------------------------------------------------
# 4. 主网络 SFCINet
# -----------------------------------------------------------
class SFCINet(nn.Module):
    def __init__(self):
        super(SFCINet, self).__init__()
        self.rgb_net = mobile_vit_small()
        self.depth_net = mobile_vit_small()

        self.r_proj1 = nn.Conv2d(160, 64, kernel_size=1) 
        self.r_proj2 = nn.Conv2d(128, 64, kernel_size=1) 
        self.r_proj3 = nn.Conv2d(96,  64, kernel_size=1) 
        self.r_proj4 = nn.Conv2d(64,  64, kernel_size=1) 

        self.d_proj1 = nn.Conv2d(160, 64, kernel_size=1)
        self.d_proj2 = nn.Conv2d(128, 64, kernel_size=1)
        self.d_proj3 = nn.Conv2d(96,  64, kernel_size=1)
        self.d_proj4 = nn.Conv2d(64,  64, kernel_size=1)

        self.fusion1 = SFS(64)
        self.fusion2 = SFS(64)
        self.fusion3 = SFS(64)
        self.fusion4 = SFS(64)

        self.decoder_1 = CHD(64)
        self.decoder_2 = CHD(64)
        self.decoder_3 = CHD(64)
        self.decoder_4 = CHD(64)

        self.head1 = nn.Conv2d(64, 1, kernel_size=1)
        self.head2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head3 = nn.Conv2d(64, 1, kernel_size=1)
        self.head4 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, rgb, depth):
        r_list = self.rgb_net(rgb)[1:5]
        d_list = self.depth_net(depth)[1:5]

        r1 = self.r_proj1(r_list[3])
        d1 = self.d_proj1(d_list[3])
        re1, de1, se1 = self.fusion1(r1, d1)
        x1 = self.decoder_1(re1, de1, se1, p_prev=None)

        r2 = self.r_proj2(r_list[2])
        d2 = self.d_proj2(d_list[2])
        re2, de2, se2 = self.fusion2(r2, d2)
        x2 = self.decoder_2(re2, de2, se2, p_prev=x1)

        r3 = self.r_proj3(r_list[1])
        d3 = self.d_proj3(d_list[1])
        re3, de3, se3 = self.fusion3(r3, d3)
        x3 = self.decoder_3(re3, de3, se3, p_prev=x2)

        r4 = self.r_proj4(r_list[0])
        d4 = self.d_proj4(d_list[0])
        re4, de4, se4 = self.fusion4(r4, d4)
        x4 = self.decoder_4(re4, de4, se4, p_prev=x3)

        return self.head4(x4), self.head3(x3), self.head2(x2), self.head1(x1)

    # -----------------------------------------------------------
    # 【极致硬核加载逻辑】：精确打印每一个权重的去向
    # -----------------------------------------------------------
    def load_pre(self, r_path, d_path):
        def _strict_load(net_name, net_module, path):
            if not path or not os.path.exists(path):
                print(f"❌ 严重警告: {net_name} 骨干预训练文件未找到！路径: {path}")
                return
            
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            # 获取严格匹配的结果 (missing_keys 和 unexpected_keys)
            missing, unexpected = net_module.load_state_dict(checkpoint, strict=False)
            
            print(f"\n================ {net_name} 骨干加载报告 ================")
            print(f"✅ 文件路径: {path}")
            print(f"🔍 总计载入参数块: {len(checkpoint.keys())}")
            if len(missing) == 0 and len(unexpected) == 0:
                print(f"🎯 状态: 【100% 完美匹配】无任何参数丢失！")
            else:
                print(f"⚠️ 状态: 包含不匹配项。")
                print(f"   -> 模型缺少参数数量 (Missing keys): {len(missing)}")
                print(f"   -> 文件多余参数数量 (Unexpected keys): {len(unexpected)}")
                # 取前 3 个打印出来看看到底丢了啥，防止满屏日志
                if missing: print(f"   [示例 Missing]: {missing[:3]}")
                if unexpected: print(f"   [示例 Unexpected]: {unexpected[:3]}")
            print("========================================================\n")

        _strict_load("RGB 支路", self.rgb_net, r_path)
        _strict_load("Depth 支路", self.depth_net, d_path)


# -----------------------------------------------------------
# 5. 损失函数
# -----------------------------------------------------------
def total_loss(preds, gt, edge=None):
    w_map = 1 + 5 * torch.abs(F.avg_pool2d(gt, kernel_size=15, stride=1, padding=7) - gt)

    def structure_loss(p, g, w):
        p_up = F.interpolate(p, size=g.shape[2:], mode='bilinear', align_corners=False)
        wbce = (w * F.binary_cross_entropy_with_logits(p_up, g, reduction='none')).sum(dim=(2,3)) / (w.sum(dim=(2,3)) + 1e-8)
        p_sig = torch.sigmoid(p_up)
        wiou = 1 - (((p_sig * g) * w).sum(dim=(2,3)) + 1) / (((p_sig + g - p_sig * g) * w).sum(dim=(2,3)) + 1)
        uncertainty_loss = 0.5 * torch.mean(1 - torch.abs(p_sig * 2 - 1)) 
        return (wbce + wiou).mean() + uncertainty_loss
    
    scale_weights = [1.0, 0.8, 0.6, 0.4] 
    loss = 0
    for sw, p in zip(scale_weights, preds):
        loss += sw * structure_loss(p, gt, w_map)
    return loss