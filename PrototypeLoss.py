# PrototypeLoss.py
# 类原型对比损失 (Class Prototype Contrastive Loss)
# 直接解决：类内差距大 + 类间差距小 的核心问题
# 推理阶段此模块完全不参与 forward，零额外开销。

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeContrastiveLoss(nn.Module):
    """
    类原型对比损失模块

    核心思想：
        1. 用 GT 标签，在 mini-batch 内为每个类别计算特征均值 → 类原型 P_c
        2. 对每个采样像素，用 InfoNCE 损失：
              拉近该像素特征 与 对应类原型
              推远该像素特征 与 其他类原型
        3. 辅以 EMA (指数移动平均) 更新全局原型，避免 batch 内样本不均匀导致原型不稳

    参数：
        num_classes  : 分割类别数（含背景）
        feat_dim     : 输入特征的通道数 (即 decode_channels)
        proj_dim     : 对比学习投影空间维度，建议 64 或 128
        temperature  : InfoNCE 温度系数，越小决策边界越硬，建议 0.07
        ignore_index : 标签中需要忽略的索引 (通常为 255)
        max_pixels   : 每次计算时随机采样的最大像素数，防止显存爆炸
        ema_momentum : EMA 全局原型的更新动量，建议 0.9~0.99
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        proj_dim: int = 64,
        temperature: float = 0.07,
        ignore_index: int = 255,
        max_pixels: int = 2048,
        ema_momentum: float = 0.99,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.temperature  = temperature
        self.ignore_index = ignore_index
        self.max_pixels   = max_pixels
        self.ema_momentum = ema_momentum

        # ── 投影头 ──────────────────────────────────────────────────────────
        # 作用：将 decode_channels 维的特征投影到更紧凑的对比空间
        # 使用 2 层 Conv1x1 + BN + ReLU，轻量且稳定
        self.projector = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, proj_dim, kernel_size=1, bias=False),
        )

        # ── EMA 全局原型库 ────────────────────────────────────────────────────
        # register_buffer: 不参与梯度计算，但会随模型 save/load
        # 初始化为全零，首个 batch 后立即用 batch 原型覆盖
        self.register_buffer(
            "proto_bank",                         # [num_classes, proj_dim]
            torch.zeros(num_classes, proj_dim)
        )
        self.register_buffer(
            "proto_initialized",                  # [num_classes] bool
            torch.zeros(num_classes, dtype=torch.bool)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 内部工具函数
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_proto_bank(self, z_flat: torch.Tensor, labels_flat: torch.Tensor):
        """
        用当前 batch 的原型 EMA 更新全局原型库。
        z_flat     : [N', proj_dim]  已 L2 归一化
        labels_flat: [N']            像素标签 (已过滤 ignore)
        """
        for c in range(self.num_classes):
            mask_c = labels_flat == c
            if mask_c.sum() == 0:
                continue
            batch_proto_c = F.normalize(z_flat[mask_c].mean(dim=0), dim=0)
            if self.proto_initialized[c]:
                # EMA 更新
                self.proto_bank[c] = (
                    self.ema_momentum * self.proto_bank[c]
                    + (1 - self.ema_momentum) * batch_proto_c
                )
                self.proto_bank[c] = F.normalize(self.proto_bank[c], dim=0)
            else:
                # 首次：直接赋值
                self.proto_bank[c] = batch_proto_c
                self.proto_initialized[c] = True

    def _compute_batch_protos(
        self,
        z_flat: torch.Tensor,
        labels_flat: torch.Tensor,
    ):
        """
        计算当前 batch 内各类的原型及其类别索引。
        返回:
            proto_matrix  : [K, proj_dim]   K 个有效类的原型
            valid_classes : [K]             对应类别列表
        """
        protos, valid_cls = [], []
        for c in range(self.num_classes):
            mask_c = labels_flat == c
            if mask_c.sum() < 4:           # 少于 4 个像素的类跳过，原型不可靠
                continue
            p = F.normalize(z_flat[mask_c].mean(dim=0), dim=0)
            protos.append(p)
            valid_cls.append(c)
        if len(protos) == 0:
            return None, None
        return torch.stack(protos, dim=0), valid_cls

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            features : [B, feat_dim, H', W']  PEPM 之后、seghead 之前的特征图
            labels   : [B, H, W]              原始分辨率 GT 标签 (LongTensor)
        返回:
            scalar loss
        """
        # ── Step 1: 投影 ──────────────────────────────────────────────────────
        z = self.projector(features)          # [B, proj_dim, H', W']
        B, D, Hf, Wf = z.shape

        # ── Step 2: 下采样标签到特征图大小 ────────────────────────────────────
        # 用 nearest 保证标签语义不被插值破坏
        labels_down = F.interpolate(
            labels.float().unsqueeze(1),
            size=(Hf, Wf),
            mode="nearest",
        ).squeeze(1).long()                   # [B, Hf, Wf]

        # ── Step 3: 展平 + 过滤无效像素 ───────────────────────────────────────
        z_flat      = z.permute(0, 2, 3, 1).reshape(-1, D)   # [N, D]
        labels_flat = labels_down.reshape(-1)                  # [N]

        valid_mask  = labels_flat != self.ignore_index
        z_flat      = z_flat[valid_mask]
        labels_flat = labels_flat[valid_mask]

        if z_flat.shape[0] < self.num_classes:
            # 像素极少，跳过本次计算（通常不会发生）
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # ── Step 4: L2 归一化 ─────────────────────────────────────────────────
        z_flat = F.normalize(z_flat, dim=1)

        # ── Step 5: 更新 EMA 全局原型库 ──────────────────────────────────────
        self._update_proto_bank(z_flat, labels_flat)

        # ── Step 6: 融合 batch 原型 + EMA 原型 ────────────────────────────────
        # 策略：优先用 batch 原型（反映当前数据分布），
        #        对 batch 中未出现的类，用 EMA 原型库补充，防止损失退化
        batch_proto_matrix, valid_cls = self._compute_batch_protos(z_flat, labels_flat)

        if batch_proto_matrix is None or len(valid_cls) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # 尝试用 EMA 原型补充 batch 中缺失的类
        ema_supplement_protos, ema_supplement_cls = [], []
        for c in range(self.num_classes):
            if c not in valid_cls and self.proto_initialized[c]:
                ema_supplement_protos.append(self.proto_bank[c])
                ema_supplement_cls.append(c)

        if len(ema_supplement_protos) > 0:
            ema_matrix = torch.stack(ema_supplement_protos, dim=0).detach()
            proto_matrix = torch.cat([batch_proto_matrix, ema_matrix], dim=0)
            all_valid_cls = valid_cls + ema_supplement_cls
        else:
            proto_matrix = batch_proto_matrix
            all_valid_cls = valid_cls

        # ── Step 7: 像素采样（防止显存爆炸）─────────────────────────────────
        # 重要：只对 batch 中存在真实原型的类别的像素计算损失
        # 保证每类采样数量均衡（防止多数类主导损失）
        z_sample_list, label_sample_list = [], []
        per_class_quota = max(64, self.max_pixels // len(valid_cls))

        for c in valid_cls:
            mask_c = labels_flat == c
            idx_c  = mask_c.nonzero(as_tuple=False).squeeze(1)
            if idx_c.shape[0] > per_class_quota:
                perm  = torch.randperm(idx_c.shape[0], device=idx_c.device)
                idx_c = idx_c[perm[:per_class_quota]]
            z_sample_list.append(z_flat[idx_c])
            label_sample_list.append(labels_flat[idx_c])

        z_sample      = torch.cat(z_sample_list,     dim=0)   # [N_s, D]
        labels_sample = torch.cat(label_sample_list, dim=0)   # [N_s]

        # ── Step 8: InfoNCE Loss ──────────────────────────────────────────────
        # sim: [N_s, K_all]   像素特征 与 所有原型 的余弦相似度
        sim = torch.mm(z_sample, proto_matrix.T) / self.temperature  # [N_s, K_all]

        # 构造正样本 mask: 像素标签 == 该列对应的类别
        all_valid_cls_t = torch.tensor(
            all_valid_cls, dtype=torch.long, device=features.device
        )
        pos_mask = (
            labels_sample.unsqueeze(1) == all_valid_cls_t.unsqueeze(0)
        ).float()                                               # [N_s, K_all]

        # 只对确实有对应正原型的像素计算损失
        has_pos = pos_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        sim        = sim[has_pos]
        pos_mask   = pos_mask[has_pos]

        log_probs  = F.log_softmax(sim, dim=1)                # [N_valid, K_all]
        # 对正样本位置取均值（一个像素只有 1 个正原型，所以 sum ≈ mean）
        loss       = -(log_probs * pos_mask).sum(dim=1).mean()

        return loss
