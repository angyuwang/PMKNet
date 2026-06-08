# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import timm

# from .MDAF import MDAF
# from .HyMKA import HyMKA



# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
#             norm_layer(out_channels),
#             nn.ReLU6()
#         )


# class ConvBN(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBN, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
#             norm_layer(out_channels)
#         )


# class Conv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
#         super(Conv, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
#         )


# class SeparableConvBNReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
#                  norm_layer=nn.BatchNorm2d):
#         super(SeparableConvBNReLU, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             norm_layer(out_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.ReLU6()
#         )


# class SeparableConvBN(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
#                  norm_layer=nn.BatchNorm2d):
#         super(SeparableConvBN, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             norm_layer(out_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         )


# class SeparableConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
#         super(SeparableConv, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         )



# class WF(nn.Module):
#     def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
#         super(WF, self).__init__()
#         self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

#         self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.eps = eps
#         self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

#     def forward(self, x, res):

#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
#         x = self.post_conv(x)
#         return x

# class WS(nn.Module):
#     def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
#         super(WS, self).__init__()
#         self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
#         self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
#         self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.eps = eps
#         self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

#     def forward(self, x, res,ade):
#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x + fuse_weights[2]*ade
#         x = self.post_conv(x)
#         return x



# class AuxHead(nn.Module):

#     def __init__(self, in_channels=64, num_classes=8):
#         super().__init__()
#         self.conv = ConvBNReLU(in_channels, in_channels)
#         self.drop = nn.Dropout(0.1)
#         self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

#     def forward(self, x, h, w):
#         feat = self.conv(x)
#         feat = self.drop(feat)
#         feat = self.conv_out(feat)
#         feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
#         return feat




# class PMKNet(nn.Module):
#     def __init__(self,
#                  decode_channels=96,
#                  dropout=0.1,
#                  # backbone_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
#                  backbone_name="convnext_tiny.in12k_ft_in1k_384",
#                  pretrained=True,
#                  window_size=8,
#                  num_classes=6,
#                  use_aux_loss = True
#                  ):
#         super().__init__()
#         self.use_aux_loss = use_aux_loss
#         self.backbone = timm.create_model(model_name=backbone_name, features_only=True,pretrained=pretrained, output_stride=32, out_indices=(0, 1, 2,3))

#         self.conv2 = ConvBN(192,decode_channels,kernel_size=1)
#         self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
#         self.conv4 = ConvBN(768, decode_channels, kernel_size=1)

#         self.MDAF_L = MDAF(decode_channels,num_heads=8,LayerNorm_type = 'WithBias')
#         self.MDAF_H = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
#         self.fuseFeature = HyMKA(in_ch=3*decode_channels, out_ch=decode_channels,num_heads=8,window_size=window_size)
#         self.WF1 = WF(in_channels=decode_channels,decode_channels=decode_channels)
#         self.WF2 = WF(in_channels=decode_channels,decode_channels=decode_channels)


#         self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
#                                                nn.Dropout2d(p=dropout, inplace=True),
#                                                Conv(decode_channels, num_classes, kernel_size=1))
#         self.down = Conv(in_channels=3*decode_channels,out_channels=decode_channels,kernel_size=1)
#     def forward(self, x,imagename=None):
#         b = x.size()[0]
#         h, w = x.size()[-2:]

#         res1,res2,res3,res4 = self.backbone(x)
#         res1h,res1w = res1.size()[-2:]

#         res2 = self.conv2(res2)
#         res3 = self.conv3(res3)
#         res4 = self.conv4(res4)
#         res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         middleres =torch.cat([res2,res3,res4],dim=1)

#         fusefeature_L,fusefeature_H,glb,local = self.fuseFeature(middleres,imagename)
#         glb = self.MDAF_L(fusefeature_L,glb)
#         local = self.MDAF_H(fusefeature_H,local)


#         res  = self.WF1(glb,local)

#         middleres = self.down(middleres)
#         res = F.interpolate(res,size=(res1h,res1w), mode='bicubic', align_corners=False)
#         res = middleres + res
#         res = self.WF2(res,res1)
#         res = self.segmentation_head(res)

#         if self.training:
#             if self.use_aux_loss == True:
#                 x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
#                 return x
#             else:
#                 x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
#                 return x
#         else:
#             x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
#             return x



# # 4.5最好
# # 双流融合模块 CFA (Hierarchical Feature Fusion)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import timm
# # [修改] 引入 CFA，替换 MDAF
# from .CFA import CFA 
# # from .MDAF import MDAF
# from .FMS3 import HyMKA

# # ==================================================================
# # 基础组件 (ConvBNReLU, WF, WS, AuxHead 等) 保持不变
# # ==================================================================

# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
#             norm_layer(out_channels),
#             nn.ReLU6()
#         )

# class ConvBN(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBN, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
#             norm_layer(out_channels)
#         )

# class Conv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
#         super(Conv, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
#         )

# class SeparableConvBNReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
#                  norm_layer=nn.BatchNorm2d):
#         super(SeparableConvBNReLU, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             norm_layer(out_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.ReLU6()
#         )

# class SeparableConvBN(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
#                  norm_layer=nn.BatchNorm2d):
#         super(SeparableConvBN, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             norm_layer(out_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         )

# class SeparableConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
#         super(SeparableConv, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         )

# class WF(nn.Module):
#     def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
#         super(WF, self).__init__()
#         self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

#         self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.eps = eps
#         self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

#     def forward(self, x, res):
#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
#         x = self.post_conv(x)
#         return x

# class WS(nn.Module):
#     def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
#         super(WS, self).__init__()
#         self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
#         self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
#         self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.eps = eps
#         self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

#     def forward(self, x, res, ade):
#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x + fuse_weights[2]*ade
#         x = self.post_conv(x)
#         return x

# class AuxHead(nn.Module):
#     def __init__(self, in_channels=64, num_classes=8):
#         super().__init__()
#         self.conv = ConvBNReLU(in_channels, in_channels)
#         self.drop = nn.Dropout(0.1)
#         self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

#     def forward(self, x, h, w):
#         feat = self.conv(x)
#         feat = self.drop(feat)
#         feat = self.conv_out(feat)
#         feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
#         return feat

# # ==================================================================
# # 主网络 PMKNet (集成 CFA)
# # ==================================================================

# class PMKNet(nn.Module):
#     def __init__(self,
#                  decode_channels=96,
#                  dropout=0.1,
#                  # backbone_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
#                  backbone_name="convnext_tiny.in12k_ft_in1k_384",
#                  pretrained=True,
#                  window_size=8,
#                  num_classes=6,
#                  use_aux_loss=True
#                  ):
#         super().__init__()
#         self.use_aux_loss = use_aux_loss
#         self.backbone = timm.create_model(model_name=backbone_name, features_only=True, pretrained=pretrained, output_stride=32, out_indices=(0, 1, 2, 3))

#         self.conv2 = ConvBN(192, decode_channels, kernel_size=1)
#         self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
#         self.conv4 = ConvBN(768, decode_channels, kernel_size=1)

#         # [修改] 移除旧的 MDAF 模块
#         # self.MDAF_L = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
#         # self.MDAF_H = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        
#         # HyMKA 保持不变，负责特征提取和初步增强
#         self.fuseFeature = HyMKA(in_ch=3*decode_channels, out_ch=decode_channels, num_heads=8, window_size=window_size)
        
#         # [修改] 引入 CFA 模块进行统一融合
#         # CFA 接收 HyMKA 的 4 个输出，内部进行 Hierarchical Fusion，输出单一特征
#         self.fusion_module = CFA(dim_in=decode_channels, dim_out=decode_channels)

#         # [修改] 移除 WF1 (因为 CFA 已经完成了融合)
#         # self.WF1 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        
#         # WF2 保留，用于融合浅层特征 (res1)
#         self.WF2 = WF(in_channels=decode_channels, decode_channels=decode_channels)

#         self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
#                                                nn.Dropout2d(p=dropout, inplace=True),
#                                                Conv(decode_channels, num_classes, kernel_size=1))
#         self.down = Conv(in_channels=3*decode_channels, out_channels=decode_channels, kernel_size=1)

#     def forward(self, x, imagename=None):
#         b = x.size()[0]
#         h, w = x.size()[-2:]

#         res1, res2, res3, res4 = self.backbone(x)
#         res1h, res1w = res1.size()[-2:]

#         res2 = self.conv2(res2)
#         res3 = self.conv3(res3)
#         res4 = self.conv4(res4)
#         res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         middleres = torch.cat([res2, res3, res4], dim=1)

#         # 1. 通过 HyMKA 提取四路特征 (含 Gated Edge Logic)
#         fusefeature_L, fusefeature_H, glb, local = self.fuseFeature(middleres, imagename)
        
#         # [注意] 如果你已经加上了 GatedEdgeModule (双流)，HyMKA 会返回 5 个值
#         # 如果 HyMKA 返回 5 个值 (含 edge_logits)，请取消下面这行的注释并注释掉上一行
#         # fusefeature_L, fusefeature_H, glb, local, edge_logits = self.fuseFeature(middleres, imagename)

#         # 2. [修改] 使用 CFA 替代 MDAF 进行融合
#         # CFA 内部会自动处理 (High+Local) 和 (Low+Global) 的配对
#         res = self.fusion_module(fusefeature_L, fusefeature_H, glb, local)

#         # 3. 后续解码流程
#         middleres = self.down(middleres)
#         res = F.interpolate(res, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res = middleres + res
        
#         # 融合浅层特征
#         res = self.WF2(res, res1)
#         res = self.segmentation_head(res)

#         if self.training:
#             if self.use_aux_loss == True:
#                 x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
#                 # 如果有 edge_logits，可以在这里一起返回
#                 return x
#             else:
#                 x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
#                 return x
#         else:
#             x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
#             return x



# # 新的边缘
# # PMKNet.py

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm

# from .CFA import CFA 
# from .HyMKA import HyMKA
# # [修改 1]: 引入我们新编写的 PEPM 边缘条件引导模块
# from .PEPM import PEPM 

# # ==================================================================
# # 基础组件保持不变 (ConvBNReLU, WF, WS, AuxHead 等)
# # ==================================================================

# class ConvBNReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBNReLU, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
#             norm_layer(out_channels),
#             nn.ReLU6()
#         )

# class ConvBN(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
#         super(ConvBN, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
#             norm_layer(out_channels)
#         )

# class Conv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
#         super(Conv, self).__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
#                       dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
#         )

# class SeparableConvBNReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
#                  norm_layer=nn.BatchNorm2d):
#         super(SeparableConvBNReLU, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             norm_layer(out_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.ReLU6()
#         )

# class SeparableConvBN(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
#                  norm_layer=nn.BatchNorm2d):
#         super(SeparableConvBN, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             norm_layer(out_channels),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         )

# class SeparableConv(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
#         super(SeparableConv, self).__init__(
#             nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
#                       padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
#                       groups=in_channels, bias=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         )

# class WF(nn.Module):
#     def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
#         super(WF, self).__init__()
#         self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
#         self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.eps = eps
#         self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

#     def forward(self, x, res):
#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
#         x = self.post_conv(x)
#         return x

# class WS(nn.Module):
#     def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
#         super(WS, self).__init__()
#         self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
#         self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
#         self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.eps = eps
#         self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

#     def forward(self, x, res, ade):
#         weights = nn.ReLU()(self.weights)
#         fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
#         x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x + fuse_weights[2]*ade
#         x = self.post_conv(x)
#         return x

# class AuxHead(nn.Module):
#     def __init__(self, in_channels=64, num_classes=8):
#         super().__init__()
#         self.conv = ConvBNReLU(in_channels, in_channels)
#         self.drop = nn.Dropout(0.1)
#         self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

#     def forward(self, x, h, w):
#         feat = self.conv(x)
#         feat = self.drop(feat)
#         feat = self.conv_out(feat)
#         feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
#         return feat

# # ==================================================================
# # 主网络 PMKNet (集成 CFA 与 PEPM)
# # ==================================================================

# class PMKNet(nn.Module):
#     def __init__(self,
#                  decode_channels=96,
#                  dropout=0.1,
#                  backbone_name="convnext_tiny.in12k_ft_in1k_384",
#                  pretrained=True,
#                  window_size=8,
#                  num_classes=6,
#                  use_aux_loss=True
#                  ):
#         super().__init__()
#         self.use_aux_loss = use_aux_loss
#         self.backbone = timm.create_model(model_name=backbone_name, features_only=True, pretrained=pretrained, output_stride=32, out_indices=(0, 1, 2, 3))

#         self.conv2 = ConvBN(192, decode_channels, kernel_size=1)
#         self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
#         self.conv4 = ConvBN(768, decode_channels, kernel_size=1)
        
#         # HyMKA 负责特征提取和初步增强 (波小与注意力融合)
#         self.fuseFeature = HyMKA(in_ch=3*decode_channels, out_ch=decode_channels, num_heads=8, window_size=window_size)
        
#         # CFA 模块进行统一融合
#         self.fusion_module = CFA(dim_in=decode_channels, dim_out=decode_channels)

#         # [修改 2]: 移除原来的 WF2 浅层特征融合模块
#         # self.WF2 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        
#         # [修改 3]: 引入 PEPM 模块
#         # dim_low = 96 是 ConvNeXt-Tiny stage0 (即 res1) 的输出通道数
#         self.ecm = PEPM(dim_low=96, dim_high=decode_channels, dim_out=decode_channels)

#         self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
#                                                nn.Dropout2d(p=dropout, inplace=True),
#                                                Conv(decode_channels, num_classes, kernel_size=1))
#         self.down = Conv(in_channels=3*decode_channels, out_channels=decode_channels, kernel_size=1)

#     def forward(self, x, imagename=None):
#         b = x.size()[0]
#         h, w = x.size()[-2:]

#         # 提取主干特征 (res1 包含最丰富的高频物理边缘)
#         res1, res2, res3, res4 = self.backbone(x)
#         res1h, res1w = res1.size()[-2:]

#         # 深层特征降维与上采样对齐
#         res2 = self.conv2(res2)
#         res3 = self.conv3(res3)
#         res4 = self.conv4(res4)
#         res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         middleres = torch.cat([res2, res3, res4], dim=1)

#         # 1. 通过 HyMKA 提取四路特征
#         fusefeature_L, fusefeature_H, glb, local = self.fuseFeature(middleres, imagename)
        
#         # 2. 使用 CFA 进行深层特征融合
#         res = self.fusion_module(fusefeature_L, fusefeature_H, glb, local)

#         # 3. 跨层残差融合
#         middleres = self.down(middleres)
#         res = F.interpolate(res, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res = middleres + res
        
#         # [修改 4]: 用 PEPM 替换旧的 WF2
#         # res1 作为 feat_low 提供物理线稿 (Query)
#         # res 作为 feat_high 提供大局语义 (Key/Value)，并在内部被提纯净化
#         res = self.ecm(feat_low=res1, feat_high=res)
        
#         # 4. 输出预测头
#         res = self.segmentation_head(res)

#         if self.training:
#             if self.use_aux_loss == True:
#                 x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
#                 return x
#             else:
#                 x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
#                 return x
#         else:
#             x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
#             return x




# 新的框架/PMKNet.py

# PMKNet.py  ── 完整版（集成类原型对比损失）
# 修改点汇总：
#   1. 新增 import PrototypeContrastiveLoss
#   2. PMKNet.__init__ 新增 self.contrast_loss
#   3. PMKNet.forward 接收可选参数 labels，训练时返回 (pred, contrast_loss_value)
#   4. 推理阶段行为与原版完全一致，无额外开销

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .HyMKA       import HyMKA
from .CFA         import CFA
from .PEPM        import PEPM
from .PrototypeLoss import PrototypeContrastiveLoss   # ← 新增导入


# ===========================================================================
# 基础卷积块（保持原版不变）
# ===========================================================================

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights  = nn.Parameter(torch.ones(2, dtype=torch.float32),
                                     requires_grad=True)
        self.eps       = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights      = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class WS(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WS, self).__init__()
        self.pre_conv  = Conv(in_channels, in_channels, kernel_size=1)
        self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
        self.weights   = nn.Parameter(torch.ones(3, dtype=torch.float32),
                                      requires_grad=True)
        self.eps       = eps
        self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

    def forward(self, x, res, ade):
        weights      = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = (fuse_weights[0] * self.pre_conv(res)
             + fuse_weights[1] * x
             + fuse_weights[2] * ade)
        x = self.post_conv(x)
        return x


class AuxHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv     = ConvBNReLU(in_channels, in_channels)
        self.drop     = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


# ===========================================================================
# 主网络 PMKNet  ── 集成类原型对比损失
# ===========================================================================

class PMKNet(nn.Module):
    """
    PMKNet with Class Prototype Contrastive Loss

    【修改说明】
    ┌──────────────────────────────────────────────────────────────────────┐
    │  新增参数:                                                            │
    │    contrast_proj_dim  : 对比投影空间维度，默认 64                     │
    │    contrast_temperature: InfoNCE 温度，默认 0.07                      │
    │    contrast_weight    : 对比损失系数 λ，默认 0.1                      │
    │    ignore_index       : 标签中忽略的索引，默认 255                    │
    │                                                                      │
    │  Forward 接口变化:                                                    │
    │    训练: forward(x, labels) → (pred, contrast_loss_value)            │
    │    推理: forward(x)         → pred          (与原版完全一致)          │
    │                                                                      │
    │  训练循环修改 (仅需改 4 行):                                          │
    │    old: output = model(images)                                       │
    │         loss = ce_loss(output, labels)                               │
    │                                                                      │
    │    new: output, loss_ct = model(images, labels)                      │
    │         loss = ce_loss(output, labels) + loss_ct                     │
    └──────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        decode_channels: int  = 96,
        dropout: float        = 0.1,
        backbone_name: str    = "convnext_tiny.in12k_ft_in1k_384",
        pretrained: bool      = True,
        window_size: int      = 8,
        num_classes: int      = 6,
        use_aux_loss: bool    = True,
        # ── 对比损失超参（新增）────────────────────────────────────────────
        contrast_proj_dim: int    = 64,
        contrast_temperature: float = 0.07,
        contrast_weight: float    = 0.1,
        contrast_ema_momentum: float = 0.99,
        ignore_index: int         = 255,
    ):
        super().__init__()
        self.use_aux_loss      = use_aux_loss
        self.contrast_weight   = contrast_weight
        self.ignore_index      = ignore_index

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            model_name=backbone_name,
            features_only=True,
            pretrained=pretrained,
            output_stride=32,
            out_indices=(0, 1, 2, 3),
        )

        # ── 深层特征降维 ──────────────────────────────────────────────────────
        self.conv2 = ConvBN(192, decode_channels, kernel_size=1)
        self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
        self.conv4 = ConvBN(768, decode_channels, kernel_size=1)

        # ── HyMKA: 物理小波 + Mamba 全局 + SPP 局部 ─────────────────────────
        self.fuseFeature = HyMKA(
            in_ch=3 * decode_channels,
            out_ch=decode_channels,
            num_heads=8,
            window_size=window_size,
        )

        # ── CFA: 跨频段融合（低频语义 ↔ 高频细节）──────────────────────────
        self.fusion_module = CFA(dim_in=decode_channels, dim_out=decode_channels)

        # ── PEPM: 物理边缘条件引导融合 ───────────────────────────────────────
        # dim_low=96 = ConvNeXt-Tiny stage-0 的输出通道数
        self.ecm = PEPM(
            dim_low=96,
            dim_high=decode_channels,
            dim_out=decode_channels,
        )

        # ── 分割预测头 ────────────────────────────────────────────────────────
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1),
        )

        self.down = Conv(
            in_channels=3 * decode_channels,
            out_channels=decode_channels,
            kernel_size=1,
        )

        # ── 类原型对比损失（新增）────────────────────────────────────────────
        # 作用于 PEPM 输出特征，训练时计算，推理时跳过
        self.contrast_loss = PrototypeContrastiveLoss(
            num_classes=num_classes,
            feat_dim=decode_channels,
            proj_dim=contrast_proj_dim,
            temperature=contrast_temperature,
            ignore_index=ignore_index,
            max_pixels=2048,
            ema_momentum=contrast_ema_momentum,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x, labels=None, imagename=None):
        """
        参数:
            x       : [B, 3, H, W]  输入图像
            labels  : [B, H, W]     GT 标签，训练时传入，推理时为 None
                      ★ 传入 labels 才会激活对比损失分支

        返回:
            训练且传入 labels: (pred [B, C, H, W],  contrast_loss scalar)
            推理 / 未传入 labels: pred [B, C, H, W]
        """
        b    = x.size(0)
        h, w = x.size()[-2:]

        # ── Step 1: Backbone 提取多尺度特征 ──────────────────────────────────
        res1, res2, res3, res4 = self.backbone(x)
        res1h, res1w = res1.size()[-2:]

        # ── Step 2: 深层特征对齐 → middleres ─────────────────────────────────
        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        middleres = torch.cat([res2, res3, res4], dim=1)   # [B, 3C, H/4, W/4]

        # ── Step 3: HyMKA 四路特征提取 ───────────────────────────────────────
        # yL   : 低频结构信息
        # yH   : 高频边缘信息 (LH + HL + HH 融合)
        # glb  : 全局语义 (Mamba)
        # local: 局部纹理 (SPP)
        fusefeature_L, fusefeature_H, glb, local = self.fuseFeature(middleres, imagename)

        # ── Step 4: CFA 跨频段融合 ────────────────────────────────────────────
        # DetailFusion:  local ←Cross-Attn→ yH   (高频引导纹理)
        # SemanticFusion: yL  ←Cross-Attn→ glb   (全局引导结构)
        res = self.fusion_module(fusefeature_L, fusefeature_H, glb, local)

        # ── Step 5: 跨层残差融合 ──────────────────────────────────────────────
        middleres_down = self.down(middleres)                      # [B, C, H/4, W/4]
        res = F.interpolate(res, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res = middleres_down + res

        # ── Step 6: PEPM 物理边缘条件引导 ────────────────────────────────────
        # res1  : 浅层高分辨率特征 → 提供物理边界线稿 (Wavelet Query)
        # res   : 深层语义特征   → 被边缘提纯 (Key / Value)
        res_feat = self.ecm(feat_low=res1, feat_high=res)          # [B, C, H/4, W/4]

        # ── Step 7: 分割预测 ──────────────────────────────────────────────────
        pred = self.segmentation_head(res_feat)
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)

        # ── Step 8: 训练时计算对比损失 ────────────────────────────────────────
        if self.training and labels is not None:
            # res_feat: [B, decode_channels, H/4, W/4]  ← 在 seghead 之前，保留足够信息
            # labels  : [B, H, W]                       ← 内部会自动下采样对齐
            loss_ct = self.contrast_loss(res_feat, labels) * self.contrast_weight
            return pred, loss_ct

        return pred


# ===========================================================================
# 训练循环修改示例  (仅供参考，不需要作为模块导入)
# ===========================================================================
"""
# ─────────────────────── 旧训练循环 ────────────────────────
for images, labels in dataloader:
    optimizer.zero_grad()

    output = model(images)                               # 旧接口
    loss   = criterion(output, labels)

    loss.backward()
    optimizer.step()


# ─────────────────────── 新训练循环 ────────────────────────
for images, labels in dataloader:
    optimizer.zero_grad()

    output, loss_ct = model(images, labels=labels)       # ← 只改这一行
    loss = criterion(output, labels) + loss_ct           # ← 加上对比损失项

    loss.backward()
    optimizer.step()


# ─── 推理/验证 loop 无需任何改动 ────────────────────────────
model.eval()
with torch.no_grad():
    output = model(images)                               # 推理接口不变
    pred   = output.argmax(dim=1)


# ─── 超参调节建议 ────────────────────────────────────────────
# contrast_weight      : 从 0.05 开始，若 Grass/Wasteland IoU 明显提升可升到 0.15
#                        若整体 mIoU 下降则调回 0.05
# contrast_temperature : 0.07（默认），类别边界清晰时可降到 0.05
# contrast_proj_dim    : 默认 64，计算资源充足时可升到 128
# contrast_ema_momentum: 0.99（默认），训练不稳定时可降到 0.9
"""
