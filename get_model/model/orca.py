import torch
import torch.nn as nn

class OrcaResidualBlock(nn.Module):
    """A single residual block with dilated convolutions"""
    def __init__(self, dilation=1, dropout=0, with_relu=True):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(64, 32, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True) if with_relu else nn.Identity(),
            nn.Conv2d(32, 64, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True) if with_relu else nn.Identity()
        )

    def forward(self, x):
        return self.conv_block(x) + x

class OrcaDecoder(nn.Module):
    def __init__(self, upsample_mode='nearest'):
        super().__init__()
        
        # Define dilation rates pattern
        dilations = [1, 2, 4, 8, 16, 32, 64] * 3

        # Create residual blocks with different dilation rates
        self.lconvtwos = nn.ModuleList([
            OrcaResidualBlock(dilation=d, dropout=0.1 if i == 0 else 0, with_relu=False)
            for i, d in enumerate(dilations)
        ])

        self.convtwos = nn.ModuleList([
            OrcaResidualBlock(dilation=d)
            for d in dilations
        ])

        # Final output layers
        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=1),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=1)
        )

        # Optional upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)

        # Combiners for different input types
        self.lcombiner = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv2d(65, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.combiner = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Distance encoding combiners
        self.lcombinerD = nn.Sequential(
            nn.Conv2d(129, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        self.combinerD = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, distenc, y=None):
        # Create interaction matrix and combine with distance encoding
        mat = x[:, :, :, None] + x[:, :, None, :]
        mat = torch.cat([mat, distenc], axis=1)
        mat = self.lcombinerD(mat)
        mat = self.combinerD(mat) + mat

        # Handle optional upsampled input
        if y is not None:
            mat = torch.cat([mat, self.upsample(y)], axis=1)
            
        cur = mat
        first = True

        # Process through residual blocks
        for lm, m in zip(self.lconvtwos, self.convtwos):
            if first:
                if y is not None:
                    cur = self.lcombiner(cur)
                    cur = self.combiner(cur) + cur
                else:
                    cur = lm(cur)
                    cur = m(cur) + cur
                first = False
            else:
                lout = lm(cur)
                if lout.size() == cur.size():
                    cur = lout + cur
                else:
                    cur = lout
                cur = m(cur) + cur

        # Final processing
        cur = self.final(cur)
        
        # Ensure output symmetry
        return 0.5 * (cur + cur.transpose(2, 3))


class OrcaDecoder_1m(nn.Module):
    """Simplified 1Mb OrcaDecoder for pretraining"""
    def __init__(self):
        super().__init__()
        
        dilations = [1, 2, 4, 8, 16, 32, 64] * 3
        
        # Initial layer processes 128 channels instead of 64
        first_block = OrcaResidualBlock(dilation=1, dropout=0.1)
        first_block.conv_block[1] = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        
        self.lconvtwos = nn.ModuleList([first_block] + [
            OrcaResidualBlock(dilation=d)
            for d in dilations[1:]
        ])

        self.convtwos = nn.ModuleList([
            OrcaResidualBlock(dilation=d)
            for d in dilations
        ])

        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=1),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=1)
        )

    def forward(self, x):
        mat = x[:, :, :, None] + x[:, :, None, :]
        cur = mat
        first = True

        for lm, m in zip(self.lconvtwos, self.convtwos):
            if first:
                cur = lm(cur)
                cur = m(cur) + cur
                first = False
            else:
                lout = lm(cur)
                if lout.size() == cur.size():
                    cur = lout + cur
                else:
                    cur = lout
                cur = m(cur) + cur

        cur = self.final(cur)
        return 0.5 * (cur + cur.transpose(2, 3))
    
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import numpy as np

class ConvBlock(nn.Module):
    """Base convolutional block with optional pooling and channel adjustment"""
    def __init__(self, in_channels, out_channels, pool_size=None, with_relu=True):
        super().__init__()
        layers = []
        if pool_size:
            layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            
        layers.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True) if with_relu else nn.Identity(),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True) if with_relu else nn.Identity()
        ])
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    """Base resolution encoder (sequence to 4kb resolution)"""
    def __init__(self):
        super().__init__()
        
        # Architecture configuration
        self.stages = [
            (4, 64, 4),    # input → 64 channels, pool by 4
            (64, 96, 4),   # 64 → 96 channels, pool by 4
            (96, 128, 4),  # 96 → 128 channels, pool by 4
            (128, 128, 5), # 128 → 128 channels, pool by 5
            (128, 128, 5), # pool by 5
            (128, 128, 5), # pool by 5
            (128, 128, 2), # pool by 2
        ]
        
        # Create encoder layers
        self.encoder_layers = nn.ModuleList()
        for i, (in_ch, out_ch, pool) in enumerate(self.stages):
            self.encoder_layers.append(nn.ModuleList([
                ConvBlock(in_ch, out_ch, pool, with_relu=False),  # lconv
                ConvBlock(out_ch, out_ch, None, with_relu=True)   # conv
            ]))

    def forward(self, x):
        """Forward propagation with gradient checkpointing"""
        binsize = 4000
        x_padding = 112000
        x_block = Blocksize  # From global configuration
        
        def process_block(x, dummy):
            cur = x
            for lconv, conv in self.encoder_layers:
                lout = lconv(cur)
                out = conv(lout)
                cur = out + lout
            return cur
            
        # Setup gradient checkpointing
        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        
        # Process in blocks to handle memory constraints
        segments = []
        starts = np.arange(0, x.size(2), x_block)
        
        for start in starts:
            if start == starts[0]:
                # First segment
                seg = x[:, :, start:start + x_block + x_padding]
                out = checkpoint(process_block, seg, dummy)
                segments.append(out[:, :, :int(x_block / binsize)])
            elif start == starts[-1]:
                # Last segment
                seg = x[:, :, start - x_padding:]
                out = checkpoint(process_block, seg, dummy)
                segments.append(out[:, :, int(x_padding / binsize):])
            else:
                # Middle segments
                seg = x[:, :, start - x_padding:start + x_block + x_padding]
                out = checkpoint(process_block, seg, dummy)
                segments.append(out[:, :, int(x_padding / binsize):int((x_block + x_padding) / binsize)])
        
        return torch.cat(segments, 2)

class BaseResBlock(nn.Module):
    """Base resolution block for Encoder2"""
    def __init__(self, channels=128, pool_size=None):
        super().__init__()
        layers = []
        if pool_size:
            layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
        layers.extend([
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(channels)
        ])
        self.lblock = nn.Sequential(*layers)
        
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        lout = self.lblock(x)
        out = self.block(lout)
        return out + lout

class Encoder2(nn.Module):
    """Multi-resolution encoder (4kb to 128kb resolution)"""
    def __init__(self):
        super().__init__()
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList([
            BaseResBlock(pool_size=2) for _ in range(5)
        ])
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList([
            BaseResBlock() for _ in range(5)
        ])
        
        # Upsample layers
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 128, kernel_size=9, padding=4),
                nn.BatchNorm1d(128),
                nn.Conv1d(128, 128, kernel_size=9, padding=4),
                nn.BatchNorm1d(128)
            ) for _ in range(5)
        ])

    def forward(self, x):
        # Encoder path
        encodings = [x]
        cur = x
        for block in self.encoder_blocks:
            cur = block(cur)
            encodings.append(cur)
            
        # Decoder path
        decodings = [cur]
        for enc, up_block, dec_block in zip(
            reversed(encodings[:-1]), 
            self.upsample_blocks,
            self.decoder_blocks
        ):
            cur = up_block(cur)
            cur = dec_block(cur)
            cur = cur + enc
            decodings.append(cur)
            
        return list(reversed(decodings))

class Encoder3(nn.Module):
    """High resolution encoder (128kb to 1024kb resolution)"""
    def __init__(self):
        super().__init__()
        
        # Similar structure to Encoder2 but with 3 resolution levels
        self.encoder_blocks = nn.ModuleList([
            BaseResBlock(pool_size=2) for _ in range(3)
        ])
        
        self.decoder_blocks = nn.ModuleList([
            BaseResBlock() for _ in range(3)
        ])
        
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv1d(128, 128, kernel_size=9, padding=4),
                nn.BatchNorm1d(128),
                nn.Conv1d(128, 128, kernel_size=9, padding=4),
                nn.BatchNorm1d(128)
            ) for _ in range(3)
        ])

    def forward(self, x):
        # Encoder path
        encodings = [x]
        cur = x
        for block in self.encoder_blocks:
            cur = block(cur)
            encodings.append(cur)
            
        # Decoder path
        decodings = [cur]
        for enc, up_block, dec_block in zip(
            reversed(encodings[:-1]), 
            self.upsample_blocks,
            self.decoder_blocks
        ):
            cur = up_block(cur)
            cur = dec_block(cur)
            cur = cur + enc
            decodings.append(cur)
            
        return list(reversed(decodings))
    
