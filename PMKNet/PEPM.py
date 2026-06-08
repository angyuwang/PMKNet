# PEPM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WaveletEdgeExtractor(nn.Module):
    """
    物理派边缘提取器：利用固定权重的 Haar 小波核，显式提取高频边界。
    不需要学习，保留最纯粹的物理边缘响应。
    """
    def __init__(self, in_channels):
        super(WaveletEdgeExtractor, self).__init__()
        # 输出通道数为输入通道数的 3 倍 (LH水平边缘, HL垂直边缘, HH对角线边缘)
        self.out_channels = in_channels * 3
        
        # 使用分组卷积实现独立通道的小波提取，stride=1 保持分辨率不变
        self.wavelet_conv = nn.Conv2d(
            in_channels, self.out_channels, 
            kernel_size=2, stride=1, padding=1, # padding=1 结合后面截取，保证大小一致
            groups=in_channels, bias=False
        )
        
        # 冻结权重，使其成为一个纯粹的物理算子
        self.init_haar_weights()
        self.wavelet_conv.weight.requires_grad = False

    def init_haar_weights(self):
        # 定义 Haar 小波的高频基底 (LH, HL, HH)
        with torch.no_grad():
            haar_weights = torch.zeros(3, 1, 2, 2)
            # LH (水平边缘)
            haar_weights[0, 0, 0, 0] = 0.5;  haar_weights[0, 0, 0, 1] = 0.5
            haar_weights[0, 0, 1, 0] = -0.5; haar_weights[0, 0, 1, 1] = -0.5
            # HL (垂直边缘)
            haar_weights[1, 0, 0, 0] = 0.5;  haar_weights[1, 0, 0, 1] = -0.5
            haar_weights[1, 0, 1, 0] = 0.5;  haar_weights[1, 0, 1, 1] = -0.5
            # HH (对角线边缘)
            haar_weights[2, 0, 0, 0] = 0.5;  haar_weights[2, 0, 0, 1] = -0.5
            haar_weights[2, 0, 1, 0] = -0.5; haar_weights[2, 0, 1, 1] = 0.5
            
            # 将基底复制到所有通道上
            in_ch = self.wavelet_conv.in_channels
            full_weights = haar_weights.repeat(in_ch, 1, 1, 1)
            self.wavelet_conv.weight.data = full_weights

    def forward(self, x):
        H, W = x.shape[2:]
        # 过小波卷积，并裁剪掉因为 padding 多出来的右下角像素，保持原分辨率
        high_freq_edges = self.wavelet_conv(x)[:, :, :H, :W]
        return high_freq_edges


class EdgeGuidedAttention(nn.Module):
    """
    边缘引导交叉注意力：将提取出的物理边缘作为 Query，去提纯语义特征。
    采用 O(N) 复杂度的轻量化点乘相似度，防止高分辨率特征显存爆炸。
    """
    def __init__(self, edge_dim, semantic_dim, hidden_dim):
        super(EdgeGuidedAttention, self).__init__()
        self.scale = math.sqrt(hidden_dim)
        
        # 1. 备料：特征映射
        self.q_proj = nn.Conv2d(edge_dim, hidden_dim, kernel_size=1)     # 边缘特征 -> Q
        self.k_proj = nn.Conv2d(semantic_dim, hidden_dim, kernel_size=1) # 语义特征 -> K
        self.v_proj = nn.Conv2d(semantic_dim, hidden_dim, kernel_size=1) # 语义特征 -> V
        
        # 2. 空间注意力门控机制
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge_feat, semantic_feat):
        # 生成 Q, K, V
        q = self.q_proj(edge_feat)      # [B, hidden_dim, H, W]
        k = self.k_proj(semantic_feat)  # [B, hidden_dim, H, W]
        v = self.v_proj(semantic_feat)  # [B, hidden_dim, H, W]
        
        # 计算空间边缘-语义匹配度 (Attention Map)
        # 将 Q 和 K 在通道维度做点乘相加，生成 1 通道的空间掩码
        attn_map = torch.sum(q * k, dim=1, keepdim=True) / self.scale
        attn_mask = self.sigmoid(attn_map) # [B, 1, H, W]
        
        # 使用边缘掩码去提纯 V (语义特征)
        # 如果某个像素有边缘响应，且语义也对得上，权重就高；否则压制噪声
        filtered_semantic = v * attn_mask
        
        return filtered_semantic, attn_mask


class PEPM(nn.Module):
    """
    主模块：Edge-Conditioned Module (边缘条件引导融合模块)
    用于替换普通的跳跃连接融合 (如 WF)，放在解码器深浅层交汇处。
    """
    def __init__(self, dim_low, dim_high, dim_out):
        super(PEPM, self).__init__()
        
        # 1. 边缘提取器
        self.edge_extractor = WaveletEdgeExtractor(dim_low)
        edge_dim = dim_low * 3 # Haar 提取出三个方向的高频
        
        # 2. 边缘引导注意力
        self.edge_attn = EdgeGuidedAttention(edge_dim=edge_dim, semantic_dim=dim_high, hidden_dim=dim_out)
        
        # 3. 语义特征初始降维
        self.semantic_proj = nn.Sequential(
            nn.Conv2d(dim_high, dim_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_out)
        )
        
        # 4. 融合与残差输出
        # 将 "提纯后的语义特征" 与 "包含浅层细节的原始特征" 缝合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim_out + dim_low, dim_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_low, feat_high):
        """
        feat_low: 浅层高分辨率特征 (如 res1) -> 用于提供真实边界
        feat_high: 深层上采样的语义特征 (如解码后的 res) -> 需要被提纯的特征
        """
        # 0. 尺寸对齐 (以浅层高分辨率为基准)
        if feat_high.shape[2:] != feat_low.shape[2:]:
            feat_high = F.interpolate(feat_high, size=feat_low.shape[2:], mode='bilinear', align_corners=False)
            
        # 1. 提取物理边缘 (线稿)
        edges = self.edge_extractor(feat_low)
        
        # 2. 语义交叉提纯 (过滤水彩画)
        feat_high_proj = self.semantic_proj(feat_high)
        filtered_semantic, _ = self.edge_attn(edges, feat_high_proj)
        
        # 3. 终极融合 (将纯净的语义 与 浅层的空间细节缝合)
        out = self.fusion(torch.cat([filtered_semantic, feat_low], dim=1))
        
        # 4. 残差连接 (保持主干梯度的顺畅)
        return out + feat_high_proj