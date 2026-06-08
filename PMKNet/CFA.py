
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x

# === 借鉴 AFSIM 的 Cross-Attention 机制 ===
class FrequencySpatialInteraction(nn.Module):
    def __init__(self, dim):
        super(FrequencySpatialInteraction, self).__init__()
        # Q: 来自辅助流 (如 yH 边缘 或 glb 全局)
        self.query_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)
        # K, V: 来自主特征流 (如 local 纹理 或 yL 结构)
        self.key_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_main, x_aux):
        """
        x_main: 主特征 (local 或 yL)
        x_aux:  辅助特征 (yH 或 glb) - 用来做 Query 指导
        """
        B, C, H, W = x_main.size()
        
        # 1. Generate Attention Map
        # Use Aux (e.g., Edge) to query Main (e.g., Texture)
        proj_query = self.query_conv(x_aux).view(B, -1, W * H).permute(0, 2, 1) # B, N, C'
        proj_key = self.key_conv(x_main).view(B, -1, W * H) # B, C', N
        
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy) # B, N, N
        
        # 2. Aggregation
        proj_value = self.value_conv(x_main).view(B, -1, W * H) # B, C, N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # 3. Residual Connection
        out = self.gamma * out + x_main
        return out

# === 组 A: 细节融合 (Detail Fusion) ===
class DetailFusion(nn.Module):
    def __init__(self, dim):
        super(DetailFusion, self).__init__()
        # 使用 Cross-Attention: 用 yH (边缘) 指导 local (纹理)
        self.cross_attn = FrequencySpatialInteraction(dim)
        self.fusion = BasicConv(dim*2, dim, 3, padding=1)

    def forward(self, f_local, f_high):
        # AFSIM 核心：用高频特征去增强空间特征的交互
        enhanced_local = self.cross_attn(f_local, f_high)
        # 再次融合原始高频信息
        return self.fusion(torch.cat([enhanced_local, f_high], dim=1))

# === 组 B: 语义融合 (Semantic Fusion) ===
class SemanticFusion(nn.Module):
    def __init__(self, dim):
        super(SemanticFusion, self).__init__()
        # 使用 Cross-Attention: 用 glb (全局) 指导 yL (结构)
        self.cross_attn = FrequencySpatialInteraction(dim)
        self.fusion = BasicConv(dim*2, dim, 3, padding=1)

    def forward(self, f_low, f_global):
        # 尺寸对齐 (Global 通常较小，需要上采样)
        if f_global.size()[2:] != f_low.size()[2:]:
            f_global = F.interpolate(f_global, size=f_low.size()[2:], mode='bilinear', align_corners=False)
            
        enhanced_low = self.cross_attn(f_low, f_global)
        return self.fusion(torch.cat([enhanced_low, f_global], dim=1))

# === 主模块: CFA (Hierarchical Frequency Fusion) ===
class CFA(nn.Module):
    def __init__(self, dim_in=128, dim_out=128):
        super(CFA, self).__init__()  # <--- 改正：要把名字改成 CFA
        hidden_dim = dim_out
        # 投影层
        self.proj_l = BasicConv(dim_in, hidden_dim, 1)
        self.proj_h = BasicConv(dim_in, hidden_dim, 1)
        self.proj_g = BasicConv(dim_in, hidden_dim, 1)
        self.proj_m = BasicConv(dim_in, hidden_dim, 1)
        
        # 分组融合 (集成 AFSIM 交互)
        self.detail_fuse = DetailFusion(hidden_dim)
        self.semantic_fuse = SemanticFusion(hidden_dim)
        
        # 最终聚合
        self.final_fuse = nn.Sequential(
            BasicConv(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.Dropout(0.1),
            BasicConv(hidden_dim, dim_out, 1)
        )

    def forward(self, yL, yH, glb, local):
        f_l = self.proj_l(yL)
        f_h = self.proj_h(yH)
        f_g = self.proj_g(glb)
        f_m = self.proj_m(local)
        
        # Group A: Detail Flow
        feat_detail = self.detail_fuse(f_m, f_h)
        
        # Group B: Semantic Flow
        feat_semantic = self.semantic_fuse(f_l, f_g)
        
        # Aggregation
        out = self.final_fuse(torch.cat([feat_detail, feat_semantic], dim=1))
        return out


