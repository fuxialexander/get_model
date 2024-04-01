import torch
import torch.nn as nn

class PositionalNorm(nn.LayerNorm):
    """PositionalNorm is a normalization layer used for 3D image inputs that
    normalizes exclusively across the channels dimension.
    https://arxiv.org/abs/1907.04312
    """

    def forward(self, x):
        # The input is of shape (B, C, H, W). Transpose the input so that the
        # channels are pushed to the last dimension and then run the standard
        # LayerNorm layer.
        x = x.permute(0, 2, 3, 1).contiguous()
        out = super().forward(x)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

class ResBlock(nn.Module):
    """Residual block following the "bottleneck" architecture as described in
    https://arxiv.org/abs/1512.03385. See Figure 5.
    The residual blocks are defined following the "pre-activation" technique
    as described in https://arxiv.org/abs/1603.05027.

    The residual block applies a number of convolutions represented as F(x) and
    then skip connects the input to produce the output F(x)+x. The residual
    block can also be used to upscale or downscale the input by doubling or
    halving the spatial dimensions, respectively. Scaling is performed by the
    "bottleneck" layer of the block. If the residual block changes the number of
    channels, or the spatial dimensions are up- or down-scaled, then the input
    is also transformed into the desired shape for the addition operation.
    """

    def __init__(self, in_chan, out_chan, scale="same"):
        """Init a Residual block.

        Args:
            in_chan: int
                Number of channels of the input tensor.
            out_chan: int
                Number of channels of the output tensor.
            scale: string, optional
                One of ["same", "upscale", "downscale"].
                Upscale or downscale by half the spatial dimensions of the
                input tensor. Default is "same", i.e., no scaling.
        """
        super().__init__()
        assert scale in ["same", "upscale", "downscale"]
        if scale == "same":
            bottleneck = nn.Conv2d(in_chan//2, in_chan//2, kernel_size=3, padding="same")
            stride = 1
        elif scale == "downscale":
            bottleneck = nn.Conv2d(in_chan//2, in_chan//2, kernel_size=3, stride=2, padding=1)
            stride = 2
        elif scale == "upscale":
            bottleneck = nn.ConvTranspose2d(in_chan//2, in_chan//2, kernel_size=4, stride=2, padding=1)
            stride = 1

        # The residual block employs the bottleneck architecture as described
        # in Sec 4. under the paragraph "Deeper Bottleneck Architectures" of the
        # original paper introducing the ResNet architecture.
        # The block uses a stack of three layers: `1x1`, `3x3` (`4x4`), `1x1`
        # convolutions. The first `1x1` reduces (in half) the number of channels
        # before the expensive `3x3` (`4x4`) convolution. The second `1x1`
        # up-scales the channels to the requested output channel size.
        self.block = nn.Sequential(
            # 1x1 convolution
            PositionalNorm(in_chan),
            nn.ReLU(),
            nn.Conv2d(in_chan, in_chan//2, kernel_size=1),

            # 3x3 convolution if same or downscale, 4x4 transposed convolution if upscale
            PositionalNorm(in_chan//2),
            nn.ReLU(),
            bottleneck,

            # 1x1 convolution
            PositionalNorm(in_chan//2),
            nn.ReLU(),
            nn.Conv2d(in_chan//2, out_chan, kernel_size=1),
        )

        # If channels or spatial dimensions are modified then transform the
        # input into the desired shape, otherwise use a simple identity layer.
        self.id = nn.Identity()
        if in_chan != out_chan or scale == "downscale":
            # We will downscale by applying a strided `1x1` convolution.
            self.id = nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride),
            )
        if scale == "upscale":
            # We will upscale by applying a nearest-neighbor upsample.
            # Channels are again modified using a `1x1` convolution.
            self.id = nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

    def forward(self, x):
        return self.block(x) + self.id(x)

class VAEEncoder(nn.Module):
    """Encoder network used for encoding the input space into a latent space.
    The encoder maps a vector(tensor) from the input space into a distribution
    over latent space. This distribution is assumed to be Normal and is
    parametrized by mu and std.
    """

    def __init__(self, in_chan, latent_dim):
        """Init an Encoder module.

        Args:
            in_chan: int
                Number of input channels of the images.
            latent_dim: int
                Dimensionality of the latent space.
        """
        super().__init__()

        # The encoder architecture follows the design of ResNet stacking several
        # residual blocks into groups, operating on different scales of the image.
        # The first residual block from each group is responsible for downsizing
        # the image and increasing the channels.

        # Start with (B,C,640,L), here is (B,1,640,640)
        self.net = nn.Sequential(
            # Stem.
            nn.Conv2d(in_chan, 8, kernel_size=3, padding="same"),  # (B,8,640,640)

            # Body.
            ResBlock(in_chan=8, out_chan=16, scale="downscale"),   # (B,16,320,320)
            ResBlock(in_chan=16, out_chan=16),
            ResBlock(in_chan=16, out_chan=16),
            ResBlock(in_chan=16, out_chan=16),

            ResBlock(in_chan=16, out_chan=32, scale="downscale"),   # (B,32,160,160)
            ResBlock(in_chan=32, out_chan=32),
            ResBlock(in_chan=32, out_chan=32),
            ResBlock(in_chan=32, out_chan=32),

            ResBlock(in_chan=32, out_chan=64, scale="downscale"),  # (B,64,80,80)
            ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=64),

            ResBlock(in_chan=64, out_chan=128, scale="downscale"), # (B,128,40,40)
            ResBlock(in_chan=128, out_chan=128),
            ResBlock(in_chan=128, out_chan=128),
            ResBlock(in_chan=128, out_chan=128),

            ResBlock(in_chan=128, out_chan=256, scale="downscale"), # (B,256,20,20)
            ResBlock(in_chan=256, out_chan=256),
            ResBlock(in_chan=256, out_chan=256),
            ResBlock(in_chan=256, out_chan=256),

            # Head.
            PositionalNorm(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), # (B,256,1,1)
            nn.Flatten(),
            nn.Linear(256, 2*latent_dim), # (B,2*latent_dim)
        )

    def forward(self, x):
        """Forward the input through the encoder and return the parameters `mu`
        and `std` of the Normal distribution.
        """
        enc = self.net(x)
        mu, log_std = torch.chunk(enc, chunks=2, dim=-1)
        return mu, log_std


class VAEDecoder(nn.Module):
    """Decoder network used for decoding the latent space back into the input
    space. The decoder maps a vector(tensor) from the latent space into a
    distribution over the input space. This distribution is assumed to be Normal
    and is parametrized by mu and std. We will assume the std to be constant.
    """

    def __init__(self, out_chan, latent_dim, encode_output_dim):
        # encode_output_dim should be dimension of encoder result before pooling. Here (B,C,M,L): (B,1,640,640) -> (B,256,20,20) 
        # reform_dim should be (256,20,20)
        """Init an Decoder module.

        Args:
            out_chan: int
                Number of output channels of the images.
            latent_dim: int
                Dimensionality of the latent space.
        """
        super().__init__()
       

        # The decoder architecture follows the design of a reverse ResNet
        # stacking several residual blocks into groups, operating on different
        # scales of the image. The first residual block from each group is
        # responsible for up-sizing the image and reducing the channels.
        self.net = nn.Sequential(
            # Inverse head.
            nn.Linear(latent_dim, encode_output_dim[0]*encode_output_dim[1]*encode_output_dim[2]), #(B,2*latent_dim) -> (B,256*8*32)
            nn.Unflatten(dim=-1, unflattened_size=encode_output_dim),     # (B,256*20*20) -> (B,256,20,20)

            # Body.
            ResBlock(in_chan=256, out_chan=128, scale="upscale"),   # (B,128,40,40)
            ResBlock(in_chan=128, out_chan=128),
            ResBlock(in_chan=128, out_chan=128),
            ResBlock(in_chan=128, out_chan=128),

            ResBlock(in_chan=128, out_chan=64, scale="upscale"),    # (B,64,80,80)
            ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=64),
            ResBlock(in_chan=64, out_chan=64),

            ResBlock(in_chan=64, out_chan=32, scale="upscale"),     # (B,32,160,160)
            ResBlock(in_chan=32, out_chan=32),
            ResBlock(in_chan=32, out_chan=32),
            ResBlock(in_chan=32, out_chan=32),
            ResBlock(in_chan=32, out_chan=32),

            ResBlock(in_chan=32, out_chan=16, scale="upscale"),   # (B,16,320,320)
            ResBlock(in_chan=16, out_chan=16),
            ResBlock(in_chan=16, out_chan=16),
            ResBlock(in_chan=16, out_chan=16),
            ResBlock(in_chan=16, out_chan=16),
   
            ResBlock(in_chan=16, out_chan=8, scale="upscale"),   # (B,8,640,640)
            ResBlock(in_chan=8, out_chan=8),
            ResBlock(in_chan=8, out_chan=8),
            ResBlock(in_chan=8, out_chan=8),
            ResBlock(in_chan=8, out_chan=8),

            # Inverse stem.
            PositionalNorm(8),
            nn.ReLU(),
            nn.Conv2d(8, out_chan, kernel_size=3, padding="same"), # (B,1,640,640)
        )

    def forward(self, x):
        """Forward the input through the encoder and return the parameters `mu`
        and `std` of the Normal distribution. Note that we will not be learning
        the covariance matrix and instead will be using a constant identity
        matrix as the covariance matrix.
        """
        mu = self.net(x)
        log_std = torch.zeros_like(mu)
        return mu, log_std