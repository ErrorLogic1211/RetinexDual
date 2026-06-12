"""
Implementation of ESDNet for image demoireing
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torchvision
from basicsr.models.archs.arch_util import to_2tuple, trunc_normal_
from torch.autograd import Function
from basicsr.models.archs.arch_util import CustomSequential
from basicsr.models.archs.L_arch_model import EBlock, DBlock
from torch.nn.parameter import Parameter
from torch.autograd.function import once_differentiable
from pdb import set_trace as stx
from timm.layers import DropPath
from typing import Optional, Callable
from functools import partial



import numbers
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref



'''
Try to update the code to have the commented Retinex Decomposer class to replace the one from RetinexFormer
'''
# def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
#     return nn.Conv2d(in_channels=in_c,
#                     out_channels=out_c,
#                     kernel_size=k,
#                     stride=s,
#                     padding=p,dilation=dilation, groups=groups)


# class RetinexDecomposer(nn.Module):
#     def __init__(self):
#         super(RetinexDecomposer, self).__init__()
#         self.decom = nn.Sequential(
#             get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
#             nn.ReLU()
#         )

#     def forward(self, input):
#         decom = self.decom(input)
#         R = decom[:, 0:3, :, :]
#         L = decom[:, 3:4, :, :]
#         return R, L


class RetinexDecomposer(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(RetinexDecomposer, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in+1, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, 
            n_fea_middle, 
            kernel_size=5, 
            padding=2, 
            bias=True, 
            groups=n_fea_middle  # Changed from n_fea_in to n_fea_middle
        )

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        self.conv3 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        mean_c = img.mean(dim=1).unsqueeze(1)
        
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)

        L_bar = self.conv2(illu_fea)
        R_bar = self.conv3(illu_fea)

        R_eff = img * L_bar + img
        L_eff = img * R_bar + img

        return R_eff, L_eff





Conv2d = nn.Conv2d
##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')

def to_3d(x):
#    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
#    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################

# class FLA(nn.Module):
    
#     def __init__(self, img_channel=3, 
#                  width=32, 
#                  middle_blk_num_enc=2,
#                  middle_blk_num_dec=2,
#                  enc_blk_nums=[2, 3],
#                  dec_blk_nums=[3, 2],
#                  dilations=[1, 4, 9],
#                  extra_depth_wise=True):
#         super(FLA, self).__init__()
        
#         self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
#                                 bias=True)
#         self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
#                               bias=True)

#         self.encoders = nn.ModuleList()
#         self.decoders = nn.ModuleList()
#         self.middle_blks = nn.ModuleList()
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()
        
#         chan = width
#         for num in enc_blk_nums:
#             self.encoders.append(
#                 CustomSequential(
#                     *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
#                 )
#             )
#             self.downs.append(
#                 nn.Conv2d(chan, 2*chan, 2, 2)
#             )
#             chan = chan * 2

#         self.middle_blks_enc = \
#             CustomSequential(
#                 *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
#             )
#         self.middle_blks_dec = \
#             CustomSequential(
#                 *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_dec)]
#             )

#         for num in dec_blk_nums:
#             self.ups.append(
#                 nn.Sequential(
#                     nn.Conv2d(chan, chan * 2, 1, bias=False),
#                     nn.PixelShuffle(2)
#                 )
#             )
#             chan = chan // 2
#             self.decoders.append(
#                 CustomSequential(
#                     *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
#                 )
#             )
#         self.padder_size = 2 ** len(self.encoders)        
        
#         # this layer is needed for the computing of the middle loss. It isn't necessary for anything else
#         self.side_out1 = nn.Conv2d(in_channels=width * 2 ** len(self.encoders), out_channels=img_channel,
#                                   kernel_size=3, stride=1, padding=1)
#         self.side_out2 = nn.Conv2d(in_channels=width * 2 ** (len(self.encoders)-1), out_channels=img_channel,
#                                   kernel_size=3, stride=1, padding=1)
       
#     def forward(self, input, side_loss=False, use_adapter=None):

#         # _, _, H, W = input.shape

#         # input = self.check_image_size(input)
#         x = self.intro(input)
        
#         skips = []
#         for encoder, down in zip(self.encoders, self.downs):
#             x = encoder(x)
#             skips.append(x)
#             x = down(x)

#         # we apply the encoder transforms
#         x_light = self.middle_blks_enc(x)
        
        
#         out = [self.side_out1(x_light)]

#         # apply the decoder transforms
#         x = self.middle_blks_dec(x_light)
#         x = x + x_light

#         for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
#             x = up(x)
#             x = x + skip
#             x = decoder(x)
#             out.append(x)
        
        
#         out[1] = self.side_out2(out[1])


        

#         x = self.ending(x)
#         x = x + input
#         out[1] = out[1] + F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
#         out[0] = out[0] + F.interpolate(input, scale_factor=0.25, mode='bilinear', align_corners=False)

#         # out[0] = out[0][..., :H//4, :W//4]
#         # out[1] = out[1][..., :H//2, :W//2]
#         # out[2] = x[:, :, :H, :W] # we recover the original size of the image
#         return out[0] , out[1], x

class FLA(nn.Module):
    
    def __init__(self, img_channel=3, 
                 width=16, 
                 extra_depth_wise=True):
        super(FLA, self).__init__()
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending1 = nn.Conv2d(in_channels=width*2, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending2 = nn.Conv2d(in_channels=width*4, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList([
            CustomSequential(
                    EBlock(width)
                ),
            CustomSequential(
                    EBlock(width*2)
                ),
            CustomSequential(
                    EBlock(width*4)
                ),
        ])
        self.decoders = nn.ModuleList([
            CustomSequential(
                    EBlock(width*4)
                ),
            CustomSequential(
                    EBlock(width*2)
                ),
            CustomSequential(
                    EBlock(width)
                ),
        ])
        self.ups = nn.ModuleList([
                    nn.Sequential(
                    nn.Conv2d(width*4, width * 8, 1, bias=False),
                    nn.PixelShuffle(2)
                ),
                    nn.Sequential(
                    nn.Conv2d(width*2, width * 4, 1, bias=False),
                    nn.PixelShuffle(2)
                )
                    
        ])
        self.downs = nn.ModuleList([
            nn.Sequential(
                     nn.Conv2d(width, 2*width, 2, 2)
                ),
            nn.Sequential(
                    nn.Conv2d(width*2, width*4, 2,2),
                )
        ])
        
        
    
        
       
    def forward(self, input):

        
        x = self.intro(input)
        out = []
        skips = []
        
        x = self.encoders[0](x) #first encoder
        # print("level 1:", x.shape)
        skips.append(x)
        x = self.downs[0](x) #doswnsample to 1/2
        
        x = self.encoders[1](x) #second encoder
        # print("level 2:", x.shape)
        skips.append(x)
        
        x = self.downs[1](x) #downsample to 1/4
        # print("level 3:", x.shape)
        x = self.encoders[2](x) #third encoder
        x = self.decoders[0](x) #first decoder
        out.append(self.ending2(x)) #1/4 output
        # print("level 3:", x.shape)
        x = self.ups[0](x) #upsample to 1/2
        
        # print("level 2:", x.shape)
        x = x + skips[1] #skip connection in level 1/2
        x = self.decoders[1](x) #second decoder
        out.append(self.ending1(x)) #1/2 output
        x = self.ups[1](x) #upsample to 1
        x = x + skips[0] #skip connection in level 1
        x = self.decoders[2](x) #third decoder
        out.append(self.ending(x)) #1 output

        out[2] = out[2] + input
        out[1] = out[1] + F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
        out[0] = out[0] + F.interpolate(input, scale_factor=0.25, mode='bilinear', align_corners=False)
        return out[0] , out[1], out[2]

def check_image_size(x):
    _, _, h, w = x.size()
    mod_pad_h = (2**4 - h % 2**4) % 2**4
    mod_pad_w = (2**4 - w % 2**4) % 2**4
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
    return x 


class RetinexDuelSambaFusionFinalization(nn.Module):
    def __init__(self,
                 in_channels=3, 
                 out_channels=3, 
                 L_n_feat=32, 
                 R_n_feat=32,
                ):
        super(RetinexDuelSambaFusionFinalization, self).__init__()
        
        self.Decomposer = RetinexDecomposer(n_fea_middle=R_n_feat, n_fea_in=in_channels, n_fea_out=out_channels)
    
        self.R_branch = UHDM(
                 in_feature_num=in_channels,
                 en_feature_num=24,
                 en_inter_num=R_n_feat,
                 de_feature_num=32,
                 de_inter_num=R_n_feat,
                 sam_number=1,
                 )
        self.L_branch = FLA(img_channel=in_channels, width=L_n_feat)

        

    def forward(self, x):
        # Decompose the input image into reflectance and illumination
        _, _, H, W = x.shape
        x = check_image_size(x)
        # print(H,W)
        reflectance, illumination = self.Decomposer(x)
        # print(reflectance.shape , illumination.shape)
        R_1, R_2, R_3 = self.R_branch(reflectance)
        L_1, L_2, L_3 = self.L_branch(illumination)

        out_1, out_2, out_3 = R_1 * L_1 , R_2 * L_2, R_3 * L_3
        out_3 = out_3[:,:,:H,:W]
        out_2 = out_2[:,:,:H//2,:W//2]
        out_1 = out_1[:,:,:H//4,:W//4]
        return out_3, out_2, out_1

     

class UHDM(nn.Module):
    def __init__(self,
                 in_feature_num=3,
                 en_feature_num=48,
                 en_inter_num=32,
                 de_feature_num=64,
                 de_inter_num=32,
                 sam_number=1,
                 ):
        super(UHDM, self).__init__()
        self.encoder = Encoder(in_feature_num=in_feature_num, feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                               sam_number=sam_number)

    def forward(self, x):

        # _, _, H, W = x.shape
        # rate = 2 ** 5
        # pad_h = (rate - H % rate) % rate
        # pad_w = (rate - W % rate) % rate
        # if pad_h != 0 or pad_w != 0:
        #     x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")

        y_1, y_2, y_3 = self.encoder(x)
        out_1, out_2, out_3 = self.decoder(y_1, y_2, y_3)

        # out_1 = out_1[:,:,:H,:W]
        # out_2 = out_2[:,:,:H//2,:W//2]
        # out_3 = out_3[:,:,:H//4,:W//4]
        out_1 = out_1 + x
        out_2 = out_2 + F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        out_3 = out_3 + F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        return out_3, out_2, out_1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)
        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)

    def forward(self, y_1, y_2, y_3):
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1 = self.decoder_1(x_1, feat=False)

        return out_1, out_2, out_3


class Encoder(nn.Module):
    def __init__(self,in_feature_num, feature_num, inter_num, sam_number):
        super(Encoder, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3


class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super(Encoder_Level, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num,  d_list=(1, 2, 3, 2, 1), depth=2, inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Decoder_Level, self).__init__()
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), depth=2, inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)
        out = self.conv(x)
        out = F.pixel_shuffle(out, 2)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)
        

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t



class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        #  our ASE here ---
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class ASSM(nn.Module):
    def __init__(self, dim, d_state, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold

        return x


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x



class EMBEDD_IN(nn.Module):   #
    def __init__(self, n_feat,scale):
        super( EMBEDD_IN, self).__init__()

        self.embedding = nn.Sequential(nn.Conv2d(n_feat, n_feat*scale, kernel_size=scale, stride=scale, bias=False))

    def forward(self, x):
        return self.embedding(x)


class EMBEDD_OUT(nn.Module): #
    def __init__(self, n_feat,scale):
        super(EMBEDD_OUT, self).__init__()

        self.embedding = nn.Sequential(nn.Conv2d(n_feat*scale, n_feat * (scale*scale), kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(scale))

    def forward(self, x):
        return self.embedding(x)
    

class SimpleGate(nn.Module):    #
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class ffn(nn.Module):   #
    def __init__(self, num_feat, ffn_expand=2):
        super(ffn, self).__init__()

        dw_channel = num_feat * ffn_expand
        self.conv1 = nn.Conv2d(num_feat, dw_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel)
        self.conv3 = nn.Conv2d(dw_channel//2, num_feat, kernel_size=1, padding=0, stride=1)
        
        self.sg = SimpleGate()

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1)*x2
        # x = x * self.sca(x)
        x = self.conv3(x)
        return x

class AttentiveMamba(nn.Module): #
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            inner_rank: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.inner_rank = inner_rank
        self.self_attention = ASSM(dim=hidden_dim, d_state=d_state,mlp_ratio=expand, inner_rank=inner_rank, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = ffn(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)


    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = x.view(B, -1, C).contiguous()
        x = self.drop_path(self.self_attention(x,x_size,self.embeddingA)) 
        x = x.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = input*self.skip_scale + x
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x

class mambablock(nn.Module):    #
    def __init__(self, dim, n_l_blocks=1, expand=2, inner_rank= 64, d_state=16):
        super().__init__()
        scale_factor= expand
        
        self.embedding_out = EMBEDD_OUT(dim,scale_factor)
        self.embedding_in = EMBEDD_IN(dim,scale_factor)
        self.l_blk = nn.Sequential(*[AttentiveMamba(dim*scale_factor, expand=expand, inner_rank = inner_rank, d_state= d_state) for _ in range(n_l_blocks)])
        
    
    def forward(self, x):
        x=self.embedding_in(x)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        # print("mamba input shape:", x.shape)
        for l_layer in self.l_blk:
            x = l_layer(x, [h, w])
        x= rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        x=self.embedding_out(x)
        
        return x


class SAM(nn.Module):
    def __init__(self, in_channel,d_list ,depth, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        # self.basic_block = mambablock(dim=in_channel, n_l_blocks=depth[0])
        # self.basic_block_2 = mambablock(dim=in_channel, n_l_blocks=depth[1])
        # self.basic_block_4 = mambablock(dim=in_channel, n_l_blocks=depth[2])
        self.fusion = CSAF(3 * in_channel, depth)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)

        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')

        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y


class CSAF(nn.Module):
    def __init__(self, in_chnls, depth, ratio=4):
        super(CSAF, self).__init__()
        # self.squeeze = nn.AdaptiveAvgPool2d((4, 4))
        # self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        # self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        # self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)
        self.mambafusion = mambablock(dim=in_chnls, n_l_blocks=depth, expand=1)

    def forward(self, x0, x2, x4):
        #edited on Traditional CSAF
        target_size = x0.shape[2:]
        x2 = F.interpolate(x2, size=target_size, mode='bilinear', align_corners=False)
        x4 = F.interpolate(x4, size=target_size, mode='bilinear', align_corners=False)
        ########################
        
        # out0 = self.squeeze(x0)
        # out2 = self.squeeze(x2)
        # out4 = self.squeeze(x4)
        out = torch.cat([x0, x2, x4], dim=1)
        # print("concat out before fusion:", out.shape)
        out = self.mambafusion(out)  # mambablock
        # out = self.compress1(out)
        # out = F.relu(out)
        # out = self.compress2(out)
        # out = F.relu(out)
        # out = self.excitation(out)
        # out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out