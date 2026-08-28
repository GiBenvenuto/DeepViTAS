import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU(inplace=True)  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse

class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Concat(nn.Module):
    def __init__(self, indices, dimension=1, sa=True):
        super().__init__()
        self.d = dimension
        self.indices = indices
        self.spatial_att = SpatialAttention(kernel_size=3)
        self.sa = sa

    def forward(self, x):
        if self.sa:
            attention_map = self.spatial_att(x[self.indices[0]])
            x[self.indices[0]] = x[self.indices[0]] * attention_map
        return torch.cat([x[i] for i in self.indices], self.d)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size[0]
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (self.img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)                 # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)                # (B, embed_dim, N_patches)
        x = x.transpose(1, 2)           # (B, N_patches, embed_dim)

        N = x.shape[1]  # n patches
        if N != self.pos_embed.shape[1]:
            # ---- interp pos_embed ----
            pos_embed_2d = self.pos_embed.transpose(1, 2).view(
                1, self.embed_dim,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size
            )
            new_h, new_w = H // self.patch_size, W // self.patch_size
            pos_embed_resized = F.interpolate(
                pos_embed_2d, size=(new_h, new_w),
                mode='bicubic', align_corners=False
            ).flatten(2).transpose(1, 2)
            x = x + pos_embed_resized
        else:
            x = x + self.pos_embed

        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Atenção
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # skip connection

        # Feedforward
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out  # skip connection
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=16, patch_size=1, in_channels=256,
     embed_dim=256, depth=2, num_heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x  # (B, N_patches, embed_dim)

def suggest_vit_params(img_size, embed_dim):
    # patch_size
    patch_size = 1 if img_size <= 40 else 2 if img_size <= 64 else 4

    # depth
    if embed_dim <= 128:
        depth = 2
    elif embed_dim <= 256:
        depth = 4
    elif embed_dim <= 512:
        depth = 6
    else:
        depth = 8


    num_heads = max(1, embed_dim // 64)
    if embed_dim % num_heads != 0:
        for h in range(num_heads, 0, -1):
            if embed_dim % h == 0:
                num_heads = h
                break

    # mlp_dim
    mlp_dim = 4 * embed_dim

    # dropout
    dropout = 0.1

    return {
        "patch_size": patch_size,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_dim": mlp_dim,
        "dropout": dropout
    }


class DeepViTAS(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, verbose=True,
                 embed_dim=512, mode='nearest', spa=True):
        super(DeepViTAS, self).__init__()
        self.mode = mode
        self.ch = in_channels
        self.nc = out_channels
        self.embed_dim = embed_dim
        self.patch_size = 1
        self.spa = spa

        self.down = nn.Sequential(
            Conv(self.ch, 32, k=3, s=2, p=1),    # 0
            Conv(32, 64, k=3, s=2, p=1),    # 1
            C2f(64, 64),                    # 2
            Conv(64, 128, k=3, s=2, p=1),   # 3
            C2f(128, 128, n=2),             # 4
            Conv(128, 256, k=3, s=2, p=1),  # 5
            C2f(256, 256, n=2)            # 6
        )
        self.vit = None
        self._vit_initialized = False

        self.up = nn.Sequential(
            Conv(self.embed_dim, 256, k=3),            # 8
            Concat([6, 8], sa=self.spa),                                # 9
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 10
            Conv(512, 128, k=3),                            # 11
            Concat([4, 11], sa=self.spa),                                # 12
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 13
            Conv(256, 64, k=3),                            # 14
            Concat([2, 14], sa=self.spa),                                # 15
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 16
            Conv(128, 64, k=3),                             # 17
            nn.Upsample(scale_factor=2.0, mode=self.mode),  # 18
            nn.Conv2d(64, self.nc, kernel_size=1)           # 20
        )
        self.sm = nn.LogSoftmax(dim=1)

    def _init_vit(self, x, H, W):
        device = x.device
        vit_params = suggest_vit_params(min(H, W), self.embed_dim)

        self.vit = ViTEncoder(
            img_size=(H, W),
            patch_size=vit_params["patch_size"],
            in_channels=x.shape[1],
            embed_dim=self.embed_dim,
            depth=vit_params["depth"],
            num_heads=vit_params["num_heads"],
            mlp_dim=vit_params["mlp_dim"],
            dropout=vit_params["dropout"]
        ).to(device)
        self.add_module("vit", self.vit)
        self._vit_initialized = True
        return vit_params['patch_size']

    def forward(self, x, y=None):
        if y != None:
            x = torch.cat((x, y), 1)

        batch_size = x.size(0)
        outputs = []
        for layer in self.down:
            x = layer(x)
            outputs.append(x)

        H, W = x.shape[2], x.shape[3]
        if self.vit is None:
            self.patch_size = self._init_vit(x, H, W)

        x = self.vit(x)
        H_out, W_out = H // self.patch_size, W // self.patch_size
        x = x.permute(0, 2, 1).view(batch_size, self.embed_dim, H_out, W_out)
        outputs.append(x)

        for layer in self.up:
            if isinstance(layer, Concat):
                x = layer(outputs)
            else:
                x = layer(x)

            outputs.append(x)

        seg_output = self.sm(x)
        return seg_output




