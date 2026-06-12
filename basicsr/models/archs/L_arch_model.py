import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

try:
    from .arch_util import LayerNorm2d
except:
    from arch_util import LayerNorm2d


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Adapter(nn.Module):
    
    def __init__(self, c, ffn_channel = None):
        super().__init__()
        if ffn_channel:
            ffn_channel = 2
        else:
            ffn_channel = c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.depthwise = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1)

    def forward(self, input):
        
        x = self.conv1(input) + self.depthwise(input)
        x = self.conv2(x)
        
        return x

# ============================================================
# Spectral Guidance Module (SGM)
# Injects a *spatial-domain* physics guidance tensor G into
# frequency-domain features (after FFT) to modulate amplitude
# and phase.  G is NEVER converted to frequency domain.
#
# Pipeline:  freq_features + G  →  conditioned freq_features
#   1. Normalise G (instance norm for stability at high-res)
#   2. Resize G to match freq_features spatial dims
#   3. Generate per-channel scale (γ) and shift (β) from G
#      for both amplitude and phase  (FiLM-style conditioning)
#   4. Apply:  out = freq * γ + β   (element-wise)
#   5. Residual:  return freq_features + α * conditioned
# ============================================================

class SpectralGuidanceModule(nn.Module):
    """
    Lightweight spatial→spectral conditioning module.

    Takes frequency-domain features (magnitude or phase, [B,C,H,W])
    and a spatial-domain guidance tensor G ([B,C,H',W']), then
    produces FiLM-style modulation weights (scale γ, shift β) from
    G to adjust the frequency features.

    Design choices for UHD efficiency:
      - Instance-norm on G (no running stats, resolution-agnostic)
      - Depthwise-separable convolution for weight generation
      - Bottleneck ratio to limit parameter count
      - Learnable residual gate α (initialised near zero)

    Args:
        dim (int):        Channel dimension of both freq features and G.
        reduction (int):  Bottleneck reduction ratio (default 4).
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        mid = max(dim // reduction, 1)

        # --- Normalise G for stable conditioning --------------------------
        self.norm_g = nn.InstanceNorm2d(dim, affine=False, eps=1e-6)

        # --- Lightweight weight generator from G -------------------------
        #   depthwise conv  →  bottleneck 1×1  →  expand back to 2*dim
        #   (first dim channels = scale γ,  second dim channels = shift β)
        self.gen = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),  # depthwise
            nn.GELU(),
            nn.Conv2d(dim, mid, kernel_size=1, bias=False),                         # bottleneck
            nn.GELU(),
            nn.Conv2d(mid, 2 * dim, kernel_size=1, bias=True),                      # expand → γ, β
        )

        # --- Learnable residual gate  (starts near 0 for safe init) ------
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, freq_features, guidance):
        """
        Args:
            freq_features : [B, C, H_f, W_f]  — magnitude or phase after FFT
            guidance      : [B, C, H_g, W_g]  — spatial-domain physics prior G
        Returns:
            conditioned   : [B, C, H_f, W_f]  — modulated frequency features
        """
        _, _, Hf, Wf = freq_features.shape

        # Step 1: normalise guidance (resolution-agnostic)
        g = self.norm_g(guidance)

        # Step 2: resize G to match frequency feature spatial dims
        _, _, Hg, Wg = g.shape
        if (Hg != Hf) or (Wg != Wf):
            g = F.interpolate(g, size=(Hf, Wf), mode='bilinear', align_corners=False)

        # Step 3: generate modulation weights  γ (scale) and β (shift)
        gb = self.gen(g)                          # [B, 2C, Hf, Wf]
        gamma, beta = gb.chunk(2, dim=1)          # each [B, C, Hf, Wf]
        gamma = gamma + 1.0   # centre scale around 1 so identity at init

        # Step 4: FiLM conditioning on frequency features
        conditioned = freq_features * gamma + beta

        # Step 5: gated residual connection
        out = freq_features + self.alpha * (conditioned - freq_features)
        return out


class FreMLP(nn.Module):
    
    def __init__(self, nc, expand = 2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

        # Spectral guidance modules for amplitude and phase conditioning
        self.sgm_mag = SpectralGuidanceModule(dim=nc)
        self.sgm_pha = SpectralGuidanceModule(dim=nc)

    def forward(self, x, guidance=None):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        # --- Inject spatial guidance G into frequency features ------------
        if guidance is not None:
            mag = self.sgm_mag(mag, guidance)   # G modulates amplitude
            pha = self.sgm_pha(pha, guidance)   # G modulates phase

        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''
    def __init__(self, c, DW_Expand, dilation = 1):
        super().__init__()
        self.dw_channel = DW_Expand * c 
        
        self.branch = nn.Sequential(
                       nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation, stride=1, groups=self.dw_channel,
                                            bias=True, dilation = dilation) # the dconv
        )
    def forward(self, input):
        return self.branch(input)
    
class DBlock(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations = [1], extra_depth_wise = False):
        super().__init__()
        #we define the 2 branches
        self.dw_channel = DW_Expand * c 

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand = 1, dilation = dilation))
            
        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c 
        self.sca = nn.Sequential(
                       nn.AdaptiveAvgPool2d(1),
                       nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
                       groups=1, bias=True, dilation = 1),  
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)


#        self.adapter = Adapter(c, ffn_channel=None)
        
#        self.use_adapters = False

#    def set_use_adapters(self, use_adapters):
#        self.use_adapters = use_adapters
        
    def forward(self, inp, adapter = None):

        y = inp
        x = self.norm1(inp)
        # x = self.conv1(self.extra_conv(x))
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)
        
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        #second step
        x = self.conv4(self.norm2(y)) # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x) # size [B, C, H, W]
        x = y + x * self.gamma
        
#        if self.use_adapters:
#            return self.adapter(x)
#        else:
        return x 

class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''
    
    def __init__(self, c):
        super().__init__()
        #we define the 2 branches
        # self.dw_channel = DW_Expand * c 
        # self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True, dilation=1) if extra_depth_wise else nn.Identity() #optional extra dw
        # self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
                
        # self.branches = nn.ModuleList()
        # for dilation in dilations:
        #     self.branches.append(Branch(c, DW_Expand, dilation = dilation))
            
        # assert len(dilations) == len(self.branches)
        # self.dw_channel = DW_Expand * c 
        # self.sca = nn.Sequential(
        #                nn.AdaptiveAvgPool2d(1),
        #                nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #                groups=1, bias=True, dilation = 1),  
        # )
        # self.sg1 = SimpleGate()
        # self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True, dilation = 1)
        # second step

        # self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc = c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.conc = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)

#        self.adapter = Adapter(c, ffn_channel=None)
        
#        self.use_adapters = False

#    def set_use_adapters(self, use_adapters):
#        self.use_adapters = use_adapters

    def forward(self, inp, guidance=None):
        y = inp
        # x = self.norm1(inp)
        # x = self.conv1(self.extra_conv(x))
        # z = 0
        # for branch in self.branches:
        #     z += branch(x)
        
        # z = self.sg1(z)
        # x = self.sca(z) * z
        # x = self.conv3(x)
        # y = inp + self.beta * x

        #second step
        x_step2 = self.norm2(y) # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2, guidance=guidance) # size [B, C, H, W]
        x = y * x_freq 
        x = self.conc(x)
        x = y + x * self.gamma

#        if self.use_adapters:
#            return self.adapter(x)
#        else:
        return x 

#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    img_channel = 3
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num = 3
    dec_blks = [3, 1, 1]
    dilations = [1, 4, 9]
    extra_depth_wise = True
    
    # net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
    #                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    net  = EBlock(c = img_channel, 
                            dilations = dilations,
                            extra_depth_wise=extra_depth_wise)

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    output = net(torch.randn((4, 3, 256, 256)))
    # print('Values of EBlock:')
    print(macs, params)

    channels = 128
    resol = 32
    ksize = 5

    # net = FAC(channels=channels, ksize=ksize)
    # inp_shape = (channels, resol, resol)
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)
    # print('Values of FAC:')
    # print(macs, params)
