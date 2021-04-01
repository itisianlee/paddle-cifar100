import math
import warnings
import numpy as np
from scipy.special import erfinv

from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle import ParamAttr
from paddle.nn.initializer import Assign
from .builder import classifier


def trunc_norm_(shape, mean=0., std=1., a=-2., b=2.):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b], The distribution of values may be incorrect.", 
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    narr = np.random.uniform(2 * l - 1, 2 * u - 1, size=shape, )

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    narr = erfinv(narr)

    # Transform to proper mean, std
    narr *= (std * math.sqrt(2.))
    narr += mean

    # Clamp to ensure it's in the proper range
    np.clip(narr, a, b, out=narr)
    return narr.astype(np.float32)


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    
    B = paddle.shape(x)[0]
    ndim = len(paddle.shape(x))
    
    shape = (B,) + (1,) * (ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = random_tensor.floor()  # binarize
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Layer):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        trunc_norm = ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02))
        self.fc1 = nn.Linear(in_features, hidden_features, trunc_norm)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, trunc_norm)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self, 
                 dim,
                 num_heads=8,  
                 qkv_bias=False, 
                 qk_scale=None, 
                 attn_drop=0., 
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        trunc_norm = ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02))
        
        self.q = nn.Linear(dim, dim, trunc_norm, qkv_bias)
        self.k = nn.Linear(dim, dim, trunc_norm, qkv_bias)
        self.v = nn.Linear(dim, dim, trunc_norm, qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, trunc_norm, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, trunc_norm)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape((-1, N, 3, self.num_heads, C // self.num_heads))
        qkv = qkv.transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = paddle.scale(paddle.matmul(q, k, transpose_y=True), scale=self.scale)
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v)
        x = x.transpose((0, 2, 1, 3))
        x = x.reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Identity(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Block(nn.Layer):

    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans, embed_dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C = x.shape[:2]
        x = x.reshape((B, C, -1))
        x = x.transpose((0, 2, 1))
        return x


class VisionTransformer(nn.Layer):
    def __init__(self, 
                 img_size=32, 
                 patch_size=4, 
                 in_chans=3, 
                 num_classes=100, 
                 embed_dim=128, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1, 
                 hybrid_backbone=None, 
                 name_prefix='',
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6)):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim 
        self.depth = depth
        self.num_heads = num_heads

        if hybrid_backbone is not None:
            # TODO
            pass
            # self.patch_embed = HybridEmbed(
            #     hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                                          embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        trunc_norm = ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02))
        bias_ = ParamAttr(initializer=nn.initializer.Constant())

        self.cls_token = self.create_parameter(shape=(1, 1, embed_dim), attr=trunc_norm)
        self.pos_embed = self.create_parameter(shape=(1, num_patches + 1, embed_dim), attr=trunc_norm)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x for x in np.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.LayerList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes, trunc_norm, bias_) if num_classes > 0 else Identity()

        for m in self.sublayers():
            if isinstance(m, nn.LayerNorm):
                paddle.assign(paddle.zeros(m.bias.shape), m.bias)
                paddle.assign(paddle.ones(m.weight.shape), m.weight)  
            if isinstance(m, nn.Linear):
                paddle.assign(trunc_norm_(m.weight.shape, std=0.02), m.weight)
                if m.bias is not None:
                    paddle.assign(paddle.zeros(m.bias.shape), m.bias)
        
        paddle.assign(trunc_norm_(self.cls_token.shape, std=0.02), self.cls_token)
        paddle.assign(trunc_norm_(self.pos_embed.shape, std=0.02), self.pos_embed)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        trunc_norm = ParamAttr(initializer=nn.initializer.TruncatedNormal(std=0.02))
        self.head = nn.Linear(self.embed_dim, num_classes, trunc_norm) if num_classes > 0 else Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]
        C = x.shape[-1]

        cls_tokens = paddle.expand(self.cls_token, (B, 1, C))  
        x = paddle.concat((cls_tokens, x), axis=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@classifier.register_module()
def vit_p4_d6_h8_e128(**kwargs):
    model = VisionTransformer(patch_size=4, embed_dim=128, depth=6, num_heads=8, **kwargs)
    return model


@classifier.register_module()
def vit_p4_d6_h8_e256(**kwargs):
    model = VisionTransformer(patch_size=4, embed_dim=256, depth=6, num_heads=8, **kwargs)
    return model


@classifier.register_module()
def vit_p4_d6_h8_e512(**kwargs):
    model = VisionTransformer(patch_size=4, embed_dim=512, depth=6, num_heads=8, **kwargs)
    return model