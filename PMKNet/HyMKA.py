import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from .SppCSPC import SppCSPC  # 确保此文件存在

# =========================================================================
# Part 1: SMFANet 核心组件 (SMFA, PCFN) - 严格复现论文思路
# =========================================================================

# --- PCFN: Partial Convolution FFN [Reference: SMFANet Paper Section 3.3] ---
class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        
        # 1x1 升维
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        # 部分卷积 (只卷一部分通道，省计算量)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1) 

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x_skip = x
        x = self.act(self.conv_0(x))
        
        if self.training:
            # 训练时 split
            x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim - self.p_dim], dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1, x2], dim=1))
        else:
            # 推理时切片 (节省显存)
            x[:, :self.p_dim, :, :] = self.act(self.conv_1(x[:, :self.p_dim, :, :]))
            x = self.conv_2(x)
            
        return x + x_skip 

# --- SMFA: Self-Modulation Feature Aggregation [Reference: SMFANet Paper Section 3.2] ---
class SMFA(nn.Module):
    def __init__(self, dim):
        super(SMFA, self).__init__()
        # 通道变换
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        # LDE Branch: Local Detail Estimation (局部细节)
        # 论文用的是 3x3 DWConv，我们针对遥感改为 3x3 + 5x5 并行，增强多尺度
        self.lde_3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.lde_5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)

        # EASA Branch: Efficient Approximation of Self-Attention (全局调制)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.gelu = nn.GELU()
        
        # 调制参数 (论文公式 4: X_m = Conv(X_s + sigma^2))
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        # Split: 一半给 EASA，一半给 LDE [Equation 1]
        y, x = self.linear_0(f).chunk(2, dim=1)
        
        # === EASA Branch (Global) ===
        # 1. Downsample (论文用 stride=8，这里用 AdaptiveAvgPool 适应不同尺寸)
        x_down = F.adaptive_avg_pool2d(x, (16, 16)) 
        x_s = self.dw_conv(x_down)
        
        # 2. Variance Calculation (论文核心: 基于方差的调制)
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        
        # 3. Modulation
        modulation = self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt))
        x_l = x * F.interpolate(modulation, size=(h, w), mode='bilinear', align_corners=False)
        
        # === LDE Branch (Local) ===
        # 局部细节增强
        y_d = self.lde_3x3(y) + self.lde_5x5(y)
        
        # Fusion [Equation 2]
        return self.linear_2(x_l + y_d)

# =========================================================================
# Part 2: HighPerfWT (物理小波 + CBAM) - 你的得分王牌
# =========================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class HighPerfWT(nn.Module):
    def __init__(self, in_ch):
        super(HighPerfWT, self).__init__()
        
        # Haar Wavelet (Frozen)
        self.haar_conv = nn.Conv2d(in_ch, in_ch*4, kernel_size=2, stride=2, groups=in_ch, bias=False)
        self.init_haar_weights()
        self.haar_conv.weight.requires_grad = False 
        
        # Context Branch (Dilation=2)
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=2, dilation=2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch*4, kernel_size=3, stride=2, padding=1, bias=False) 
        )
        
        self.cbam = CBAM(in_ch*4)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def init_haar_weights(self):
        with torch.no_grad():
            haar_weights = torch.zeros(4, 1, 2, 2)
            # Haar Basis
            haar_weights[0, 0, :, :] = 0.5 
            haar_weights[1, 0, 0, 0] = 0.5; haar_weights[1, 0, 0, 1] = 0.5
            haar_weights[1, 0, 1, 0] = -0.5; haar_weights[1, 0, 1, 1] = -0.5
            haar_weights[2, 0, 0, 0] = 0.5; haar_weights[2, 0, 0, 1] = -0.5
            haar_weights[2, 0, 1, 0] = 0.5; haar_weights[2, 0, 1, 1] = -0.5
            haar_weights[3, 0, 0, 0] = 0.5; haar_weights[3, 0, 0, 1] = -0.5
            haar_weights[3, 0, 1, 0] = -0.5; haar_weights[3, 0, 1, 1] = 0.5
            
            in_ch = self.haar_conv.in_channels
            full_weights = haar_weights.repeat(in_ch, 1, 1, 1)
            self.haar_conv.weight.data = full_weights

    def forward(self, x):
        haar_feat = self.haar_conv(x)
        res_feat = self.res_branch(x)
        feat = haar_feat + self.alpha * res_feat
        feat = self.cbam(feat)
        B, C4, H, W = feat.shape
        C = C4 // 4
        out = feat.view(B, C, 4, H, W)
        yL = out[:, :, 0, :, :]      
        yH_LH = out[:, :, 1, :, :]   
        yH_HL = out[:, :, 2, :, :]   
        yH_HH = out[:, :, 3, :, :]   
        return yL, yH_LH, yH_HL, yH_HH


# =========================================================================
# Part 3: Attention Blocks (Global + Local) - 融合 SMFA 与 Dual-Branch KAN
# =========================================================================

# class GlobalAttention(nn.Module):
#     def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8, relative_pos_embedding=True):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // self.num_heads
#         self.scale = head_dim ** -0.5
#         self.ws = window_size
#         self.qkv = nn.Conv2d(dim, 3*dim, kernel_size=1, bias=qkv_bias)
#         self.proj = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=window_size, padding=window_size//2, groups=dim, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False)
#         )
#         self.attn_x = nn.Conv2d(dim,dim,kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
#         self.attn_y = nn.Conv2d(dim,dim,kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))
#         self.relative_pos_embedding = relative_pos_embedding
#         if self.relative_pos_embedding:
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
#             coords_h = torch.arange(self.ws)
#             coords_w = torch.arange(self.ws)
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
#             coords_flatten = torch.flatten(coords, 1)
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#             relative_coords[:, :, 0] += self.ws - 1
#             relative_coords[:, :, 1] += self.ws - 1
#             relative_coords[:, :, 0] *= 2 * self.ws - 1
#             relative_position_index = relative_coords.sum(-1)
#             self.register_buffer("relative_position_index", relative_position_index)
#             trunc_normal_(self.relative_position_bias_table, std=.02)

#     def pad(self, x, ps):
#         _, _, H, W = x.size()
#         if W % ps != 0: x = F.pad(x, (0, ps - W % ps), mode='reflect')
#         if H % ps != 0: x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
#         return x

#     def pad_out(self, x):
#         x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
#         return x

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.pad(x, self.ws)
#         B, C, Hp, Wp = x.shape
#         qkv = self.qkv(x)
#         q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
#                           d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
#         dots = (q @ k.transpose(-2, -1)) * self.scale
#         if self.relative_pos_embedding:
#             relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#                 self.ws * self.ws, self.ws * self.ws, -1)
#             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#             dots += relative_position_bias.unsqueeze(0)
#         attn = dots.softmax(dim=-1)
#         attn = attn @ v
#         attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
#                          d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)
#         attn = attn[:, :, :H, :W]
#         out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
#               self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
#         out = self.pad_out(out)
#         out = self.proj(out)
#         out = out[:, :, :H, :W]
#         return out


# class S2SSM(nn.Module):
#     def __init__(self, dim=256, outdim=256, num_heads=16, window_size=8, drop_path=0., norm_layer=nn.BatchNorm2d, **kwargs):
#         super().__init__()
#         self.down = nn.Conv2d(dim, outdim, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
#         self.norm_smfa = norm_layer(outdim)
#         self.smfa = SMFA(outdim)
#         self.norm1 = norm_layer(outdim)
#         self.attn = GlobalAttention(outdim, num_heads=num_heads, window_size=window_size)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(outdim)
#         self.pcfn = PCFN(outdim)

#     def forward(self, x):
#         x = self.down(x)
#         x = x + self.smfa(self.norm_smfa(x))
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.pcfn(self.norm2(x))
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

# 尝试导入官方 Mamba。如果还没装环境，先用 MockMamba 顶替，方便你测试网络维度通不通。
try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed. Using Mock linear layer for testing dimensions.")
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.proj = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.proj(x)

# =========================================================================
# 替换原有的 GlobalAttention 为 SqueezedWindowMamba
# =========================================================================
class SqueezedWindowMamba(nn.Module):
    def __init__(self, dim=256, window_size=8):
        super().__init__()
        self.dim = dim
        self.ws = window_size

        # 1. 局部窗口 Mamba (Local Scanning)
        # 负责在 window_size x window_size 内部提取细粒度的空间几何纹理
        self.local_mamba = Mamba(
            d_model=dim,
            d_state=16,  # 默认状态维度
            d_conv=4,    # 默认局部卷积核大小
            expand=2     # 块内特征扩展倍数 (为了轻量化，这里设为2，普通 Mamba 常设为4)
        )

        # 2. 全局交互 Mamba (Global Cross-window Interaction)
        # 负责处理被 "Squeeze" 后的窗口代表 Token，获取整张高分影像的宏观上下文
        self.global_mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # 保持你原有的输出投影逻辑，确保与外部特征完美对接
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=window_size, padding=window_size//2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

    # 沿用你写好的 padding 函数，这部分非常稳
    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0: x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0: x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 补齐边缘，确保能被 window_size 整除
        x_padded = self.pad(x, self.ws)
        _, _, Hp, Wp = x_padded.shape

        # ==================== Step 1: 窗口划分 (Window Partition) ====================
        # (B, C, H, W) -> (B, num_windows, window_area, C)
        x_windows = rearrange(x_padded, 'b c (nh ws1) (nw ws2) -> b (nh nw) (ws1 ws2) c', 
                              ws1=self.ws, ws2=self.ws)
        B_w, num_windows, L_w, C_w = x_windows.shape

        # ==================== Step 2: 局部窗口扫描 (Local Mamba) ====================
        # 将 Batch 和 窗口数 合并为 1D 序列维度，送入 Mamba
        x_local = x_windows.view(B_w * num_windows, L_w, C_w)
        out_local = self.local_mamba(x_local) 
        out_local = out_local.view(B_w, num_windows, L_w, C_w)

        # ==================== Step 3: 窗口挤压 (Squeeze) ====================
        # 作者的核心创新：把每个窗口压缩成 1 个代表 Token
        # squeezed_tokens 形状: (B, num_windows, C)
        squeezed_tokens = out_local.mean(dim=2) 

        # ==================== Step 4: 全局交互 (Cross-window Interaction) ====================
        # 让所有窗口代表相互通信，获取全局感受野
        global_context = self.global_mamba(squeezed_tokens) # (B, num_windows, C)

        # ==================== Step 5: 广播与特征融合 (Broadcast & Fusion) ====================
        # 把全局信息“反向拓展”回 8x8 窗口内部的每一个像素上
        global_context = global_context.unsqueeze(2).expand(-1, -1, L_w, -1)
        
        # 将精细局部纹理与宏观全局语义进行融合
        out_fused = out_local + global_context 

        # 重新排列回图像格式 (B, C, Hp, Wp)
        out = rearrange(out_fused, 'b (nh nw) (ws1 ws2) c -> b c (nh ws1) (nw ws2)', 
                        nh=Hp//self.ws, nw=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        # 截掉之前 pad 的部分，恢复原始尺寸
        out = out[:, :, :H, :W]

        # ==================== Step 6: 保持你原有的投影对齐 ====================
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# try:
#     from mamba_ssm import Mamba
# except ImportError:
#     print("Warning: mamba_ssm not installed. Using Mock linear layer.")
#     class Mamba(nn.Module):
#         def __init__(self, d_model, **kwargs):
#             super().__init__()
#             self.proj = nn.Linear(d_model, d_model)
#         def forward(self, x): return self.proj(x)

# # ==========================================
# # 优化 1: 双向 Mamba 封装 (解决空间瞎子问题)
# # ==========================================
# class BiMambaWrapper(nn.Module):
#     """专门为视觉任务设计的双向 Mamba"""
#     def __init__(self, dim, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         # 正向 Mamba
#         self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
#         # 反向 Mamba (由于序列扫描需要，通常用两套独立参数效果更好)
#         self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
#     def forward(self, x):
#         # x shape: (B, L, C)
#         out_fwd = self.mamba_fwd(x)
#         # 将序列翻转后送入，再翻转回来
#         out_bwd = self.mamba_bwd(torch.flip(x, dims=[1]))
#         out_bwd = torch.flip(out_bwd, dims=[1])
#         # 融合正反向信息
#         return out_fwd + out_bwd


# # ==========================================
# # 优化 2 & 3: 优化版 Squeezed Window Mamba
# # ==========================================
# class SqueezedWindowMamba(nn.Module):
#     def __init__(self, dim=256, window_size=8):
#         super().__init__()
#         self.dim = dim
#         self.ws = window_size

#         # 1. 局部扫描：换成双向 BiMamba，确保局部窗口内的空间感知无死角
#         self.local_mamba = BiMambaWrapper(dim=dim, d_state=16, d_conv=4, expand=2)
        
#         # 2. 全局交互：同样换成双向 BiMamba
#         self.global_mamba = BiMambaWrapper(dim=dim, d_state=16, d_conv=4, expand=2)

#         # 3. 增强型挤压 (Enhanced Squeeze): 引入一个轻量级的 MLP 或 Conv 避免高频特征丢失
#         # 这里用 1x1 卷积融合 mean 和 max 提取的 Token
#         self.squeeze_fusion = nn.Sequential(
#             nn.Linear(dim * 2, dim),
#             nn.GELU(),
#             nn.Linear(dim, dim)
#         )

#         # 4. 动态门控融合 (Adaptive Fusion Parameter)
#         # 初始化为 0，让网络前期先保留局部的稳定性，后期再慢慢吸取全局 Mamba 的信息
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))

#         self.proj = nn.Sequential(
#             # 引入一个 3x3 卷积 (Depthwise) 恢复 2D 空间归纳偏置
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False)
#         )

#     def pad(self, x, ps):
#         _, _, H, W = x.size()
#         if W % ps != 0: x = F.pad(x, (0, ps - W % ps), mode='reflect')
#         if H % ps != 0: x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
#         return x

#     def pad_out(self, x):
#         return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_padded = self.pad(x, self.ws)
#         _, _, Hp, Wp = x_padded.shape

#         # --- Step 1: Window Partition ---
#         x_windows = rearrange(x_padded, 'b c (nh ws1) (nw ws2) -> b (nh nw) (ws1 ws2) c', 
#                               ws1=self.ws, ws2=self.ws)
#         B_w, num_windows, L_w, C_w = x_windows.shape

#         # --- Step 2: Local Bi-Scanning ---
#         x_local = x_windows.view(B_w * num_windows, L_w, C_w)
#         out_local = self.local_mamba(x_local) 
#         out_local = out_local.view(B_w, num_windows, L_w, C_w)

#         # --- Step 3: Enhanced Squeeze (Avg + Max) ---
#         # 既保留背景语义(mean)，又保留显著的微弱高频变化(max)
#         token_mean = out_local.mean(dim=2) # (B, num_windows, C)
#         token_max, _ = out_local.max(dim=2) # (B, num_windows, C)
#         tokens_cat = torch.cat([token_mean, token_max], dim=-1) # (B, num_windows, 2C)
#         squeezed_tokens = self.squeeze_fusion(tokens_cat) # (B, num_windows, C)

#         # --- Step 4: Global Interaction ---
#         global_context = self.global_mamba(squeezed_tokens) # (B, num_windows, C)

#         # --- Step 5: Adaptive Broadcast & Fusion ---
#         global_context = global_context.unsqueeze(2).expand(-1, -1, L_w, -1)
        
#         # 使用 self.gamma 进行残差缩放 (极其重要！)
#         out_fused = out_local + self.gamma * global_context 

#         # --- Step 6: Reverse & Project ---
#         out = rearrange(out_fused, 'b (nh nw) (ws1 ws2) c -> b c (nh ws1) (nw ws2)', 
#                         nh=Hp//self.ws, nw=Wp//self.ws, ws1=self.ws, ws2=self.ws)

#         out = out[:, :, :H, :W]
        
#         # 经过带有 3x3 DWConv 的 proj 层，补回 2D 空间特性
#         out = self.pad_out(out)
#         out = self.proj(out)
#         out = out[:, :, :H, :W]

#         return out



class S2SSM(nn.Module):
    def __init__(self, dim=256, outdim=256, num_heads=16, window_size=8, drop_path=0., norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.down = nn.Conv2d(dim, outdim, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.norm_smfa = norm_layer(outdim)
        self.smfa = SMFA(outdim)
        self.norm1 = norm_layer(outdim)
        
        # 【修改这里】：用 SqueezedWindowMamba 替换原有的 GlobalAttention
        self.attn = SqueezedWindowMamba(dim=outdim, window_size=window_size)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(outdim)
        self.pcfn = PCFN(outdim)

    # forward 逻辑一字不改
    def forward(self, x):
        x = self.down(x)
        x = x + self.smfa(self.norm_smfa(x))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.pcfn(self.norm2(x))
        return x

# === 新增: 高效的 Depthwise KAN 模块 ===
class DepthwiseKAN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. Base Branch
        self.base_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.base_act = nn.GELU()
        
        # 2. Spline Proxy Branch
        self.spline_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.spline_act = nn.SiLU() 
        self.spline_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False) 
        
        # 3. KAN 内部动态权重
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        base = self.base_act(self.base_conv(x))
        spline = self.spline_proj(self.spline_act(self.spline_conv(x)))
        return base + self.gamma * spline


# === 修改: 双分支残差局部注意力模块 ===
class DualBranchLocalAttention(nn.Module):
    def __init__(self, dim=256, window_size=8):
        super().__init__()
        # ----------- Branch 1: Main Spatial Flow -----------
        self.local_spp = SppCSPC(dim, dim)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=window_size, padding=window_size//2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
        
        # ----------- Branch 2: Detail Non-linear Flow -----------
        self.kan_branch = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), 
            DepthwiseKAN(dim),                  
            nn.BatchNorm2d(dim)                 
        )
        
        # ----------- Dual-Branch Adaptation Fusion -----------
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.fusion_conv = nn.Conv2d(dim, dim, 1, bias=False)

    def pad_out(self, x): 
        return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        
    def forward(self, x):
        B, C, H, W = x.shape
        spp_out = self.local_spp(x)
        spp_out = self.pad_out(spp_out)
        main_out = self.proj(spp_out)[:, :, :H, :W]
        kan_out = self.kan_branch(x)
        fused = main_out + self.alpha * kan_out
        return self.fusion_conv(fused)


# === 修改: DKAN ===
class DKAN(nn.Module):
    def __init__(self, dim=256, outdim=256, window_size=8, drop_path=0., norm_layer=nn.BatchNorm2d, **kwargs):
        super().__init__()
        self.down = nn.Conv2d(dim, outdim, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.norm1 = norm_layer(outdim)
        self.attn = DualBranchLocalAttention(outdim, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(outdim)
        self.pcfn = PCFN(outdim)

    def forward(self, x):
        x = self.down(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.pcfn(self.norm2(x))
        return x

# 修改了这个了
# # =========================================================================
# # Part 3: Attention Blocks (Global + Local) - 融合 SMFA
# # =========================================================================

# class GlobalAttention(nn.Module):
#     # (保持原有的 Swin-Window Attention 逻辑，这部分代码很稳)
#     def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8, relative_pos_embedding=True):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // self.num_heads
#         self.scale = head_dim ** -0.5
#         self.ws = window_size
#         self.qkv = nn.Conv2d(dim, 3*dim, kernel_size=1, bias=qkv_bias)
#         self.proj = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=window_size, padding=window_size//2, groups=dim, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False)
#         )
#         self.attn_x = nn.Conv2d(dim,dim,kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
#         self.attn_y = nn.Conv2d(dim,dim,kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))
#         self.relative_pos_embedding = relative_pos_embedding
#         if self.relative_pos_embedding:
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
#             coords_h = torch.arange(self.ws)
#             coords_w = torch.arange(self.ws)
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
#             coords_flatten = torch.flatten(coords, 1)
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()
#             relative_coords[:, :, 0] += self.ws - 1
#             relative_coords[:, :, 1] += self.ws - 1
#             relative_coords[:, :, 0] *= 2 * self.ws - 1
#             relative_position_index = relative_coords.sum(-1)
#             self.register_buffer("relative_position_index", relative_position_index)
#             trunc_normal_(self.relative_position_bias_table, std=.02)

#     def pad(self, x, ps):
#         _, _, H, W = x.size()
#         if W % ps != 0: x = F.pad(x, (0, ps - W % ps), mode='reflect')
#         if H % ps != 0: x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
#         return x

#     def pad_out(self, x):
#         x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
#         return x

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.pad(x, self.ws)
#         B, C, Hp, Wp = x.shape
#         qkv = self.qkv(x)
#         q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
#                             d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
#         dots = (q @ k.transpose(-2, -1)) * self.scale
#         if self.relative_pos_embedding:
#             relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#                 self.ws * self.ws, self.ws * self.ws, -1)
#             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#             dots += relative_position_bias.unsqueeze(0)
#         attn = dots.softmax(dim=-1)
#         attn = attn @ v
#         attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
#                          d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)
#         attn = attn[:, :, :H, :W]
#         out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
#               self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
#         out = self.pad_out(out)
#         out = self.proj(out)
#         out = out[:, :, :H, :W]
#         return out

# class S2SSM(nn.Module):
#     def __init__(self, dim=256, outdim=256, num_heads=16, window_size=8, drop_path=0., norm_layer=nn.BatchNorm2d, **kwargs):
#         super().__init__()
#         # 1. 降维/特征变换
#         self.down = nn.Conv2d(dim, outdim, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        
#         # 2. SMFA (Pre-processing): 注入全局统计信息
#         # 这是 SMFANet 的灵魂，防止 Window Attention 坐井观天
#         self.norm_smfa = norm_layer(outdim)
#         self.smfa = SMFA(outdim)
        
#         # 3. Window Attention (Core)
#         self.norm1 = norm_layer(outdim)
#         self.attn = GlobalAttention(outdim, num_heads=num_heads, window_size=window_size)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
#         # 4. PCFN (Post-processing): 通道融合 & 去噪
#         self.norm2 = norm_layer(outdim)
#         self.pcfn = PCFN(outdim)

#     def forward(self, x):
#         x = self.down(x)
        
#         # Step 1: SMFA 增强
#         x = x + self.smfa(self.norm_smfa(x))
        
#         # Step 2: Attention 交互
#         x = x + self.drop_path(self.attn(self.norm1(x)))
        
#         # Step 3: PCFN 修正
#         x = x + self.pcfn(self.norm2(x))
        
#         return x

# class LocalAttention(nn.Module):
#     def __init__(self, dim=256, window_size=8):
#         super().__init__()
#         self.local = SppCSPC(dim, dim)
#         self.proj = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=window_size, padding=window_size//2, groups=dim, bias=False),
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, dim, kernel_size=1, bias=False)
#         )
#     def pad_out(self, x): return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
#     def forward(self, x):
#         B, C, H, W = x.shape
#         local = self.local(x)
#         out = self.pad_out(local)
#         out = self.proj(out)
#         out = out[:, :, :H, :W]
#         return out

# class DKAN(nn.Module):
#     def __init__(self, dim=256, outdim=256, window_size=8, drop_path=0., norm_layer=nn.BatchNorm2d, **kwargs):
#         super().__init__()
#         self.down = nn.Conv2d(dim, outdim, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
#         self.norm1 = norm_layer(outdim)
#         self.attn = LocalAttention(outdim, window_size=window_size)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
#         # 给 Local 分支也加上 PCFN，保证一致性
#         self.norm2 = norm_layer(outdim)
#         self.pcfn = PCFN(outdim)

#     def forward(self, x):
#         x = self.down(x)
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.pcfn(self.norm2(x))
#         return x

# =========================================================================
# Part 4: HyMKA 主模块 (整合入口)
# =========================================================================
class HyMKA(nn.Module):
    def __init__(self, in_ch, out_ch, num_heads=8, window_size=8):
        super(HyMKA, self).__init__()
        
        # 1. 物理小波基座 (HighPerfWT) - 绝对不改
        self.wt = HighPerfWT(in_ch)
        
        # 2. 全局交互 (三明治结构: SMFA-Attn-PCFN)
        self.glb = S2SSM(dim=in_ch, outdim=in_ch, num_heads=num_heads, window_size=window_size)
        
        # 3. 局部感知 (SPP-PCFN)
        self.localb = DKAN(dim=in_ch, outdim=in_ch, num_heads=8, window_size=window_size)
        
        # 4. 融合卷积 (保持不变)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.outconv_bn_relu_glb = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.outconv_bn_relu_local = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, imagename=None):
        # 1. 物理域增强
        yL, y_LH, y_HL, y_HH = self.wt(x)

        # 2. 高频融合
        yH = torch.cat([y_LH, y_HL, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)

        # 3. 投影输出
        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)

        # 4. 智能域交互 (SMFA -> Window -> PCFN)
        glb = self.outconv_bn_relu_glb(self.glb(x))
        local = self.outconv_bn_relu_local(self.localb(x))
        
        return yL, yH, glb, local