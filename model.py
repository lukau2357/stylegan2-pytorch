import torch
import numpy as np
import math
import torchvision

from typing import Tuple, List, Union
from losses import gradient_penalty, PathLengthPenalty
from utils import generate_noise, generate_style_mixes

def leaky_relu(p : float = 0.2):
    return torch.nn.LeakyReLU(negative_slope = p)

# StyleGAN2-ADA implementation uses sqrt(2) gain for LeakyReLU as well, even though it depends on chosen negative slope
# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/torch_utils/ops/bias_act.py#L23
# We've set it to actual gain, for the time being.
def get_leaky_relu_gain(p : float = 0.2):
    return (2 / (1 + p ** 2)) ** 0.5
    # return 2 ** 0.5

class EqualizedWeight(torch.nn.Module):
    """
    Equalized learning rate for all weigts in StyleGAN2 model. Authors reason that standard optimization algorithms like RMSProp or Adam
    normalize the update by an estimate of gradient's variance, and thus weights with higher variances could take more time to adapt. Instead 
    of performing weight initialization techniques, they divide the weights by Kaiming normalization constant (https://arxiv.org/pdf/1502.01852), 
    attempting to make variances/dynamic ranges (and thus learning speeds) for all weights the same. More can be found in ProGAN paper: https://arxiv.org/pdf/1710.10196. 
    """
    def __init__(self, shape : Tuple, gain : float = 1, lr_mul : float = 1):
        """
            shape - Desired shape that weights should take
        """
        super().__init__()
        self.c = gain * np.prod(shape[1:]) ** (-0.5)
        self.lr_mul = lr_mul
        self.weight = torch.nn.Parameter(torch.randn(shape) / lr_mul) # Dividing by lr_mul here and multiplying in the forward pass ensures that X has standard deviation c.

    def forward(self):
        """
        Performs self-scaling of weights with Kaiming constant.
        """
        return self.weight * self.c * self.lr_mul

class EqualizedLinear(torch.nn.Module):
    """
    Linear layer that uses an instance of EqualizedWeight for weights.
    """
    def __init__(self, in_features : int, out_features : int, bias : int = 0, gain : float = 1, lr_mul : float = 1):
        super().__init__()
        self.w = EqualizedWeight((out_features, in_features), gain = gain, lr_mul = lr_mul)
        self.b = torch.nn.Parameter(torch.ones(out_features) * bias)
        self.lr_mul = lr_mul

    def forward(self, X : torch.Tensor):
        """
        X: (batch_size, in_features, out_features). self.w() is called to enforce for reasons discussed above. 

        Return: Tensor of shape (batch_size, out_features), result of affine transformation
        """
        b = self.b * self.lr_mul
        return torch.nn.functional.linear(X, self.w(), bias = b)

class EqualizedConv2d(torch.nn.Module):
    """
    2D Covnolutional layer with EqualizedWeight filter. Biases are not subject to learning rate 'equalization'.
    """
    def __init__(self, in_channels : int, out_channels : int, kernel_size : int, stride : int = 1, padding : Union[str, int] = "same", bias : int = 0, use_bias : bool = True, gain : float = 1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias

        self.w = EqualizedWeight((out_channels, in_channels, kernel_size, kernel_size), gain = gain)
        if self.use_bias:
            self.b = torch.nn.Parameter(torch.ones(out_channels) * bias)

        else:
            self.register_buffer("b", torch.zeros(out_channels))

    def forward(self, X : torch.Tensor):
        """
        X: (batch_size, in_channels, h, w)

        Return: (batch_size, out_channels, h', w'), result of 2D convolution with equalized learning rate filters. h' and w' depend on convolution parameters,
        if padding = 'same' (which it is by default), h' = w, w' = w.
        """
        return torch.nn.functional.conv2d(X, self.w(), bias = self.b, padding = self.padding, stride = self.stride)

class MappingNetwork(torch.nn.Module):
    """
    Mapping network that attempts to disentangle z-space into w-space, as a stack of fully connected layers with LeakyReLU activation.

    In StyleGAN1, they find that higher learning rates for style mapping network can lead to unstable learning, and thus reduce learning rate for those weights (and biases)
    by two orders of magnitude. Instead of having a seperate optimizer with adjusted learning rate, they 'lazily' multiply corresponding weights and biases by the desired 
    factor of decay. Keep in mind that this is not exactly equivalent to explicitly modyfing the learning rate within the optimizer for desired parameters, but it gets
    the job done apparently.
    """
    def __init__(self, latent_dim : int, depth : int, lr_mul : float = 0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.depth = depth
        self.lr_mul = lr_mul
        layers = []

        for _ in range(depth):
            layers.extend([EqualizedLinear(latent_dim, latent_dim, gain = get_leaky_relu_gain(), lr_mul = lr_mul), leaky_relu()])

        self.net = torch.nn.Sequential(*layers)

    def forward(self, z : torch.Tensor) -> torch.Tensor:
        """
        z: (batch_size, latent_dim). 
        Return:
            w: (batch_size, latent_dim)
        """
        z = torch.nn.functional.normalize(z, dim = 1)
        return self.net(z)

class Conv2dModulation(torch.nn.Module):
    """
    Modulated 2D convolution, with optional demodulation to enforce unit variance. Modulation is perform directly on filter weights, i-th channel for 
    every filter is scaled by s_i, which comes from W-space by appling an affine transformation. Optionally, to keep unit variance for i-th channel,
    demodulation operation is introduced in the paper, and it can be derived easily by hand. Lastly, since filter weights for each sample will be different
    due to differing style scales, authors propose using grouped convolutions for an efficient implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, demodulate = True, stride = 1, padding = "same", dilation = 1, eps = 1e-8, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.weight = EqualizedWeight((out_channels, in_channels, kernel_size, kernel_size), gain = kwargs.get("gain", 1.0))
        self.eps = eps
        self.padding = padding

    def forward(self, X : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        """
        X (batch_size, in_channels, h, w) - Input feature maps
        y (batch_size, in_channels) - Per-channel modulation parameters for each sample
        Return:
            z: (batch_size, out_channels, h, w) after modulated convoliton is applied.
        """

        b, c, h, w = X.shape    
        # Weights are repeated for each batch. Then, per-channel scaling factors are applied. For a fixed sample, for all filters, and 
        # for i-th input channel, scaling factors applied to convolution filters corresponding to i-th input channel are the same.
        w1 = self.weight().unsqueeze(0) * y[:, None, :, None, None]
        
        if self.demodulate:
            d = torch.rsqrt((w1 ** 2).sum(dim = (2, 3, 4), keepdim = True) + self.eps)
            w1 = w1 * d # w1 *= d caused in place error modifications for gradient computations in pathe length regularization!!!

        res = torch.nn.functional.conv2d(X.reshape(1, -1, h, w), 
                                         w1.reshape(b * self.out_channels, c, self.kernel_size, self.kernel_size), 
                                         groups = b, stride = self.stride, dilation = self.dilation, padding = self.padding)
        
        return res.reshape((b, -1, h, w))

class StyleBlock(torch.nn.Module):
    """
    StyleBlock consists of Conv2dModulation, along with affine maps that correspond to style scaling factors. At the end, for each sample in the batch, IID Gaussian noise
    of shape (current_resolution, current_resolution) is broadcast accross all channels (https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L294)
    Additionally for StyleGAN2, a single scaling factor for the noise is used, unlike StyleGAN which usses different noise scaling factors per output channels. Also, 
    bias parameters for each output channel are learned.

    StyleGAN2-ADA implementation of this layer: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L304C22-L304C30
    """
    def __init__(self, in_channels : int, out_channels : int, latent_dim : int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        self.style_scale_map = EqualizedLinear(latent_dim, in_channels, bias = 1.)
        self.noise_bias = torch.nn.Parameter(torch.zeros((out_channels)))
        self.noise_scale = torch.nn.Parameter(torch.zeros(1))
        self.conv = Conv2dModulation(in_channels, out_channels, 3, gain = get_leaky_relu_gain())
        self.activation = leaky_relu()

    def forward(self, X : torch.Tensor, w : torch.Tensor, noise : torch.Tensor):
        """
        X: (batch_size, in_channels, h, w) - Feature maps from previous layer
        w: (batch_size, latent_dim) - Style vector w for each sample that is used to generate style scales
        noise: (batch_size, 1, h, w) - Noise broadcast to every output channel, different for every sample.
        """
        style_scales = self.style_scale_map(w)
        X = self.conv(X, style_scales)
        return self.activation(X + self.noise_bias[None, :, None, None] + (self.noise_scale * noise))

class ToRgb(torch.nn.Module):
    """
    StyleGAN2 paper is ambigous about this layer, official implementation for StyleGAN2-ADA:
    https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L310

    ToRgb is different than other convolutional layers:
        - It applies 1x1 modulated convolution with kernel size 1 to obtain desired number of channels in output (3 or 4)
        - It does not apply demodulation, like convolutions used in style blocks.
    """
    def __init__(self, latent_dim, in_channels, out_channels = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.affine = EqualizedLinear(latent_dim, in_channels)
        self.conv = Conv2dModulation(in_channels, out_channels, 1, demodulate = False)
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, X, w):
        style_scales = self.affine(w)
        rgb = self.conv(X, style_scales)
        return rgb + self.bias[None, :, None, None]
    
class GeneratorBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim

        self.gen_block_1 = StyleBlock(in_channels, out_channels, latent_dim)
        self.gen_block_2 = StyleBlock(out_channels, out_channels, latent_dim)
        self.trgb = ToRgb(latent_dim, out_channels)
    
    def forward(self, X: torch.Tensor, w: torch.Tensor, noise : Tuple[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.gen_block_1(X, w, noise[0])
        X = self.gen_block_2(X, w, noise[1])
        return X, self.trgb(X, w)

class Smooth(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Based on: https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py#L27 smoothing filter is [1, 2, 1]
        # Official StyleGan2-ADA uses [1, 3, 3, 1] (https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L132C73-L132C84)
        # Also, there is reference [64] in StyleGan paper that references usage of this kernel
        # Can't seem to get it working with [1, 3, 3, 1] though, asymmetric padding
        filter = torch.tensor([[1, 2, 1]])
        filter = filter.T.matmul(filter).detach().float()[None, None, :, :]
        filter /= filter.sum()
        # Normalize the filter
        self.register_buffer("filter", filter)
        self.pad = torch.nn.ReplicationPad2d((1)) # https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html
        
    def forward(self, X : torch.Tensor):
        b, c, h, w = X.shape
        X = X.reshape(-1, 1, h, w) 
        X = torch.nn.functional.conv2d(self.pad(X), self.filter)
        return X.reshape(b, c, h, w)

class Upsample(torch.nn.Module):
    def __init__(self, sf = 2):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor = sf, mode = "bilinear", align_corners = False)
        self.smooth = Smooth()
    
    def forward(self, X : torch.Tensor):
        return self.smooth(self.upsample(X))

class Downsample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = torch.nn.Upsample(scale_factor = 0.5, mode = "bilinear", align_corners = False)
        self.smooth = Smooth()
    
    def forward(self, X : torch.Tensor):
        return self.downsample(self.smooth(X))

class Generator(torch.nn.Module):
    def __init__(self, image_size : int, latent_dim : int, network_capacity : int = 8, max_features : int = 512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.max_features = max_features

        # Channel shrinking strategy per resolution taken from https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py
        # StyleGan papers are too ambiguous in this regard. With network_capacity = 8 and image_size = 1024, obtained number of filters per resolution
        # is consistent with ProGAN paper, Table 2 in the paper. (https://arxiv.org/pdf/1710.10196)
        self.num_layers = int(math.log2(image_size) - 1) # -1 because first 4x4 resolution layer is not considered as a full generator block
        self.filters = [min(max_features, network_capacity * (2 ** (i + 1))) for i in range(self.num_layers)][::-1]
        self.start_tensor = torch.nn.Parameter(torch.randn(1, self.filters[0], 4, 4))
        self.style_start = StyleBlock(self.filters[0], self.filters[0], latent_dim)
        self.to_rgb_start = ToRgb(latent_dim, self.filters[0])
        self.blocks = torch.nn.ModuleList([GeneratorBlock(self.filters[i - 1], self.filters[i], latent_dim) for i in range(1, len(self.filters))])
        self.upsample = Upsample()

    def forward(self, w : torch.Tensor, noise : List[Tuple[torch.Tensor]]):
        """
        w: (self.num_layers + 1, batch_size, latent_dim) - For each sample, style vector to be used at layer i. Elegantlly encompasses style mixing regularization
        """
        w = w[None, :].expand(self.num_layers + 1, -1 , -1) if len(w.shape) == 2 else w

        _, b, _ = w.shape
        # Repeath starting tensor for each sample in the batch
        X = self.style_start(self.start_tensor.expand(b, -1, -1, -1), w[0], noise[0][0])
        rgb = self.to_rgb_start(X, w[0])
    
        for i in range(1, len(noise)):
            X_new, rgb_new = self.blocks[i - 1].forward(self.upsample(X), w[i], (noise[i][0], noise[i][1]))
            rgb = rgb_new + self.upsample(rgb)
            X = X_new
        
        # Original StyleGan2 paper does not include nonlinearities when computing RGBs accross different resolutions. Therefore, output image pixels are not constrainted
        # to any range, although this can easily be done through min/max normalization per-channels. We will leave it as is, for now
        return rgb
    
class DiscriminatorBlock(torch.nn.Module):
    """
    Two convolutions with LeakyReLU in between. Skip 1x1 equalized convolution without bias is applied to the input, and residual connection is weighed by 1 / sqrt(2)
    to account for no normalization layers. https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L574

    In skip_conv, downsampling is applied first, and then 1x1 convolution takes place.
    conv2 uses strided convolutions for downsampling, and applies smoothing filter before convolution takes place.

    Links to StlyeGAN2 and StlyeGAN2-ADA official implementations from which these parameters
    were inferred:
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/torch_utils/ops/conv2d_resample.py#L119
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/torch_utils/ops/conv2d_resample.py#L106
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L552
        https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py#L653
    """
    def __init__(self, in_channels : int, out_channels : int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_conv = EqualizedConv2d(in_channels, out_channels, 1, bias = False)
        self.conv1 = EqualizedConv2d(in_channels, in_channels, 3, gain = get_leaky_relu_gain())
        self.conv2 = EqualizedConv2d(in_channels, out_channels, 3, stride = 2, padding = 1, gain = get_leaky_relu_gain())
        self.activation = leaky_relu()
        self.smooth = Smooth()
        self.ds = Downsample()

    def forward(self, X : torch.Tensor):
        """
        X: (batch_size, in_channels, h, w) input feature map
        """
        Y = self.skip_conv(self.ds(X))
        X = self.activation(self.conv1(X))
        X = self.activation(self.conv2(self.smooth(X)))
        return (X + Y) * (2 ** (-0.5))

class MiniBatchStd(torch.nn.Module):
    """
    This layer is added at the end of discriminator. It divides input batch into b / group_size groups of size group_size,
    and feature maps into groups of size c / num_channels. Then, for every subgroup of batches, and subgroup of feature maps, it computes the mean
    of standard deviations of all spatial positions. Lastly, it concatenates this information to the original input - so every spatial position for every
    sample will have "knowledge" about previously computed statistics, depending on which group the given spatial position belonged to. The idea is that this speeds
    up training of Discriminator in earlier stages. This technique was first introduced in ProGAN paper: https://arxiv.org/pdf/1710.10196, and this implementation
    follows the implementation given in StyleGAN2-ADA: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L589
    """
    def __init__(self, group_size = 4, num_channels = 1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, X : torch.Tensor):
        """
            X: (batch_size, in_channels, h, w)
        """
        N, C, H, W = X.shape
        G = min(self.group_size, N)
        F = self.num_channels

        assert N % G == 0
        assert C % F == 0
        n = N // G
        c = C // F

        y = X.reshape(G, n, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c
        y = y - y.mean(dim = 0)            # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim = 0)       # [nFcHW]  Calc variance over group. This is a biased estimate, not dividing by (G - 1) 
        y = (y + 1e-8).sqrt()              # [nFcHW]  Calc stddev over group.
        y = y.mean(dim = [2, 3, 4])        # [nF]     Take average over channels and pixels, these are "means of stdevs" computed in previous step
        y = y.reshape(-1, F, 1, 1)         # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)           # [NFHW]   Replicate over group and pixels.
        X = torch.cat([X, y], dim = 1)     # [N(C + F)HW]   Append to input as new channels.
        return X

class DiscriminatorEpilogue(torch.nn.Module):
    """
    Last layer of the discriminator. StyleGAN2 discriminator last layer seems to deviate from that of ProGAN in Table 2. For this implementation, we've consulted
    StyleGAN2-ADA implementation for this layer: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L615
    """
    def __init__(self, in_channels : int, resolution : int, use_mbstd : bool = True, mbstd_group_size : int = 4, mbstd_num_channels : int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.use_mbstd = use_mbstd
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_channels = mbstd_num_channels

        if self.use_mbstd:
            self.mbstd = MiniBatchStd(group_size = mbstd_group_size, num_channels = mbstd_num_channels)

        self.conv = EqualizedConv2d(self.in_channels + self.mbstd_num_channels if self.use_mbstd else self.in_channels, self.in_channels, 3, gain = get_leaky_relu_gain())
        self.lrelu = leaky_relu()
        self.fc = torch.nn.Sequential(
            EqualizedLinear(in_channels * resolution ** 2, in_channels, gain = get_leaky_relu_gain()),
            self.lrelu,
            EqualizedLinear(in_channels, 1)
        )
    
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """
        X: (batch_size, self.in_channels, self.resolution, self.resolution)
        """
        if self.use_mbstd:
            X = self.mbstd(X)
        
        X = self.lrelu(self.conv(X))
        return self.fc(X.flatten(1))

class Discriminator(torch.nn.Module):
    def __init__(self, 
                 input_res : int, 
                 in_channels : int, 
                 network_capacity : int = 8,
                 max_features : int = 512, 
                 use_mbstd : bool = True, 
                 mbstd_group_size : int = 4, 
                 mbstd_num_channels : int = 1):
        super().__init__()
        self.input_res = input_res
        self.in_channels = in_channels
        self.network_capacity = network_capacity
        self.max_features = max_features
        self.use_mbstd = use_mbstd
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_channels = mbstd_num_channels

        self.res_log = int(math.log2(self.input_res))
        # Use same channel shrinking/expanding strategy as with Generator. Fits Table 2 of ProGAN paper when network_capacity = 8 and input_res = 1024
        self.filters = [min(max_features, network_capacity * 2 ** (i + 1)) for i in range(self.res_log - 1)]
        self.lrelu = leaky_relu()

        # FromRGB extends the number of channels to self.filters[0] with kernel size 1, with Leaky ReLU application.
        # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py#L543
        # For now, working with RGB images only - 3 input chanels that is.
        self.from_rgb = torch.nn.Sequential(EqualizedConv2d(3, self.filters[0], 1, gain = get_leaky_relu_gain()), self.lrelu)
        self.disc_blocks = torch.nn.Sequential(*[
            DiscriminatorBlock(self.filters[i], self.filters[i + 1]) for i in range(len(self.filters) - 1)
        ])

        # TODO: Last resolution hardcoded to 4, perhaps make it dynamic somehow?
        self.last = DiscriminatorEpilogue(self.filters[-1], 4, use_mbstd = use_mbstd, mbstd_group_size = mbstd_group_size, mbstd_num_channels = mbstd_num_channels)
    
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """
        X: (batch_size, 3, h, w)
        """
        X = self.from_rgb(X)
        X = self.disc_blocks(X)
        return self.last(X)

class StyleGAN2(torch.nn.Module):
    def __init__(self, 
                 latent_dim : int, 
                 target_resolution : int, 
                 mlp_depth : int = 8, 
                 mlp_lr_multiplier : float = 0.01, 
                 network_capacity : int = 8, 
                 max_features : int = 512, 
                 gen_ema_coeff : float = 0.999,
                 use_mbstd : bool = True, 
                 mbstd_group_size : int = 4, 
                 mbstd_num_channels : int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_resolution = target_resolution
        self.mlp_depth = mlp_depth
        self.ml_lr_multiplier = mlp_lr_multiplier
        self.network_capacity = network_capacity
        self.max_features = max_features
        self.gen_ema_coeff = gen_ema_coeff
        self.use_mbstd = use_mbstd
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_channels = mbstd_num_channels

        self.MN = MappingNetwork(latent_dim, mlp_depth, lr_mul = mlp_lr_multiplier)
        self.G = Generator(target_resolution, latent_dim, network_capacity = network_capacity, max_features = max_features)
        self.D = Discriminator(target_resolution, 3, network_capacity = network_capacity, max_features = max_features, use_mbstd = use_mbstd, mbstd_group_size = mbstd_group_size, mbstd_num_channels = mbstd_num_channels)

        self.GE = Generator(target_resolution, latent_dim, network_capacity = network_capacity, max_features = max_features)
        self.MNE = MappingNetwork(latent_dim, mlp_depth, lr_mul = mlp_lr_multiplier)

    def EMA(self):
        """
        Perform exponential moving average update of Generator weights (including the mapping network), using gen_ema_coeff. 
        This was first introduced, to my knowledge at least in ProGAN paper, https://arxiv.org/pdf/1710.10196. Another paper goes into mathematical details
        of why this works: https://arxiv.org/pdf/1806.04498
        """

        for g_param, ge_param in zip(self.G.parameters(), self.GE.parameters()):
            p_new = g_param.data
            p_old = ge_param.data
            ge_param.data = self.gen_ema_coeff * p_old + (1 - self.gen_ema_coeff) * p_new
        
        for mn_param, mne_param in zip(self.MN.parameters(), self.MNE.parameters()):
            p_new = mn_param.data
            p_old = mne_param.data
            mne_param.data = self.gen_ema_coeff * p_old + (1 - self.gen_ema_coeff) * p_new

    def num_parameters(self, verbose : bool = True) -> Tuple[int, int, int]:
        gen_params, mapping_params, disc_params = 0, 0, 0
        def _count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        gen_params = _count_parameters(self.G)
        mapping_params = _count_parameters(self.MN)
        disc_params = _count_parameters(self.D)

        if verbose:
            print(f"Number of mapping network parameters: {mapping_params / 1e6:.2f}M")
            print(f"Number of generator parameters: {gen_params / 1e6:.2f}M")
            print(f"Number of discriminator/critic parameters: {disc_params / 1e6:.2f}M")
        
        return gen_params, mapping_params, disc_params

    def forward_gen(self, w : torch.Tensor, device : str) -> torch.Tensor:
        return self.G(w, generate_noise(self.target_resolution, w.shape[1], device))

if __name__ == "__main__":
    DEVICE = "cuda"
    stylegan = StyleGAN2(512, 128).to(DEVICE)
    bs = 32
    generator = stylegan.G

    z = torch.randn((32, 512)).to(DEVICE)
    w = stylegan.MN(z)
    noise = generate_noise(128, 32, DEVICE)
    X_fake = stylegan.G.forward(w, noise)
    torch.autograd.set_detect_anomaly(True)
    plp = PathLengthPenalty()
    plp.forward(w, X_fake)