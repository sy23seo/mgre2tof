import os
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

import numpy as np                                                                             
# from denoising_diffusion_pytorch.H5WindowDataset import H5WindowDataset  ## SeoSY 2025-07-24 (dataset class에서 npy불러오기 위해서서)
# from denoising_diffusion_pytorch.H5WindowDataset_cldm import H5WindowDataset_CLDM  ## SeoSY 2025-10-10 (controlnet용 dataset class)
from denoising_diffusion_pytorch.H5WindowDataset_unified_1 import H5WindowDataset, H5WindowDataset_CLDM  ## SeoSY 2025-10-13 위두개르 ㄹ통합

import contextlib
from tqdm.auto import tqdm as _tqdm

from torch.utils.data import Sampler  # ← 추가

from typing import Optional
from pathlib import Path


from torch.utils.tensorboard import SummaryWriter

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model
class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False,
        # -------------------------------------------------20251010 수정 controlnet update (옵션 추가)
        use_control: bool = False,
        cond_channels: int | None = None,
        control_inject_down: bool = True,
        control_inject_mid: bool = True,
        control_inject_up: bool = False,          # 기본 False: 논문 관행(업 비주입)
        freeze_locked: bool = True               # 메인(locked) 동결 여부
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3) # 입력크기 그대로 유지됨

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) # [(64,128), (128,256), (256,512), (512,1024)]

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages) # (False, False, False, True)
        attn_heads = cast_tuple(attn_heads, num_stages) # (4, 4, 4, 4)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)   # (32, 32, 32, 32)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers (LOCKED path = 기존 본체)
        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

        # -------------------------------------------------20251010 수정 controlnet update (컨트롤 모드 설정)
        self.use_control = bool(use_control and (cond_channels is not None))
        self.cond_channels = cond_channels
        self.control_inject_down = control_inject_down
        self.control_inject_mid  = control_inject_mid
        self.control_inject_up   = control_inject_up
        self.freeze_locked = freeze_locked

        # -------------------------------------------------20251010 수정 controlnet update (ControlNet: trainable copy + zero-conv 앞/뒤)
        if self.use_control:
            #---------------------------------------2025.10.11 ControlNet 도입부 수정
            # (A) 어떤 cond 채널(K=5/15/64...)이 와도 마지막 채널을 64(=dims[0])로 정렬하는 힌트 인코더
            self.control_hint_encoder = nn.Sequential(
                nn.Conv2d(self.cond_channels, 16, 3, padding=1), nn.SiLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.SiLU(),   # H/2
                nn.Conv2d(32, 96, 3, padding=1), nn.SiLU(),   # H/4
                nn.Conv2d(96, 256, 3, padding=1), nn.SiLU(),  # H/8
                nn.Conv2d(256, dims[0], 3, padding=1)                   # 최종 64ch
            )
            # 초기 주입 영향 0으로 시작하고 싶으면 마지막 conv를 zero-init
            nn.init.zeros_(self.control_hint_encoder[-1].weight)
            nn.init.zeros_(self.control_hint_encoder[-1].bias)
            #---------------------------------------2025.10.11 ControlNet 도입부 수정

            # (A') cond 해상도 정합용 다운 경로 (cond를 각 down-stage의 dim_in으로 이동)
            self.control_cond_downsamplers = ModuleList([])
            #---------------------------------------2025.10.11 ControlNet 도입부 수정
            # 힌트 인코더를 거치면 시작 채널은 dims[0](=64)로 고정
            c_curr = dims[0]
            #---------------------------------------2025.10.11 ControlNet 도입부 수정
            for ind, (_din, dout) in enumerate(in_out):
                is_last = ind >= (num_resolutions - 1)
                if not is_last:
                    self.control_cond_downsamplers.append(Downsample(c_curr, dout))
                else:
                    self.control_cond_downsamplers.append(nn.Conv2d(c_curr, dout, 3, padding=1))
                c_curr = dout
            self.control_cond_mid = nn.Identity()  # 미드 해상도는 위에서 맞춰져 있음

            # 업 경로용 업샘플러(옵션)
            if self.control_inject_up:
                self.control_cond_upsamplers = ModuleList([])
                rev_in_out = list(reversed(in_out))
                for ind, (din, dout) in enumerate(rev_in_out):
                    is_last = ind == (len(rev_in_out) - 1)
                    if not is_last:
                        self.control_cond_upsamplers.append(Upsample(c_curr, din))
                    else:
                        self.control_cond_upsamplers.append(nn.Conv2d(c_curr, din, 3, padding=1))
                    c_curr = din

            # (B) trainable copy (down & mid & (opt) up) + zero-conv(pre/post)
            self.ctrl_down_pre = ModuleList([])     # pre zero-conv(cond -> dim_in)
            self.ctrl_down_copy = ModuleList([])    # trainable copy blocks
            self.ctrl_down_post = ModuleList([])    # post zero-conv(output -> dim_in), y에 더함

            for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                zip(in_out, full_attn, attn_heads, attn_dim_head)
            ):
                if not self.control_inject_down:
                    break
                attn_klass = FullAttention if layer_full_attn else LinearAttention

                # pre zero-conv: cond feature를 dim_in으로 (이미 채널 일치하지만 zero-conv로 게이트)
                pre = nn.Conv2d(dim_in, dim_in, 1, bias=True)
                nn.init.zeros_(pre.weight); nn.init.zeros_(pre.bias)
                self.ctrl_down_pre.append(pre)

                # trainable copy: 원 블록 복제 (res1, res2, attn) - downsample은 복제 X
                locked_res1, locked_res2, locked_attn, _locked_down = self.downs[ind]
                copy_res1 = copy.deepcopy(locked_res1)
                copy_res2 = copy.deepcopy(locked_res2)
                copy_attn = copy.deepcopy(locked_attn)
                self.ctrl_down_copy.append(ModuleList([copy_res1, copy_res2, copy_attn]))

                # post zero-conv: copy 출력 채널을 dim_in으로 정렬해서 잔차 생성
                post = nn.Conv2d(dim_in, dim_in, 1, bias=True)
                nn.init.zeros_(post.weight); nn.init.zeros_(post.bias)
                self.ctrl_down_post.append(post)

            # mid
            if self.control_inject_mid:
                self.ctrl_mid_pre  = nn.Conv2d(mid_dim, mid_dim, 1, bias=True)
                self.ctrl_mid_post = nn.Conv2d(mid_dim, mid_dim, 1, bias=True)
                nn.init.zeros_(self.ctrl_mid_pre.weight);  nn.init.zeros_(self.ctrl_mid_pre.bias)
                nn.init.zeros_(self.ctrl_mid_post.weight); nn.init.zeros_(self.ctrl_mid_post.bias)

                self.ctrl_mid_copy = ModuleList([
                    copy.deepcopy(self.mid_block1),
                    copy.deepcopy(self.mid_attn),
                    copy.deepcopy(self.mid_block2)
                ])
            else:
                self.ctrl_mid_pre = self.ctrl_mid_post = None
                self.ctrl_mid_copy = None

            # up (옵션)
            if self.control_inject_up:
                self.ctrl_up_pre  = ModuleList([])
                self.ctrl_up_copy = ModuleList([])
                self.ctrl_up_post = ModuleList([])

                rev_in_out = list(reversed(in_out))
                for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
                    zip(reversed(in_out), reversed(full_attn), reversed(attn_heads), reversed(attn_dim_head))
                ):
                    attn_klass = FullAttention if layer_full_attn else LinearAttention

                    # up에서는 cat 후 채널이 (dim_out + dim_in)
                    pre = nn.Conv2d(dim_out + dim_in, dim_out + dim_in, 1, bias=True)
                    nn.init.zeros_(pre.weight); nn.init.zeros_(pre.bias)
                    self.ctrl_up_pre.append(pre)

                    locked_res1, locked_res2, locked_attn, _locked_up = self.ups[ind]
                    copy_res1 = copy.deepcopy(locked_res1)
                    copy_res2 = copy.deepcopy(locked_res2)
                    copy_attn = copy.deepcopy(locked_attn)
                    self.ctrl_up_copy.append(ModuleList([copy_res1, copy_res2, copy_attn]))

                    post = nn.Conv2d(dim_out, dim_out, 1, bias=True)  # block2 출력 채널 = dim_out
                    nn.init.zeros_(post.weight); nn.init.zeros_(post.bias)
                    self.ctrl_up_post.append(post)

            # (C) locked 본체 동결 옵션
            if self.freeze_locked:
                for p in self.downs.parameters(): p.requires_grad = False
                self.mid_block1.requires_grad_(False)
                for p in self.mid_attn.parameters(): p.requires_grad = False
                self.mid_block2.requires_grad_(False)
                for p in self.ups.parameters(): p.requires_grad = False
                self.final_res_block.requires_grad_(False)
                self.final_conv.requires_grad_(False)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    # -------------------------------------------------20251010 수정 controlnet update (forward에 cond 및 Control 흐름 반영)
    def forward(self, x, time, x_self_cond = None, cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        # -------- cond를 현재 해상도로 맞춰가며 보관 (down 경로 기준)
        cond_feats = []
        if self.use_control and (cond is not None):
            #---------------------------------------2025.10.11 ControlNet 도입부 수정
            # 입력 cond(K채널)를 hint encoder로 통과시켜 원해상도에서 64채널로 정렬
            h_c = self.control_hint_encoder(cond)   # (B, 64, H, W)
            cond_feats.append(h_c)                  # stage 0 (dim_in = 64)

            # 이후 각 down-stage의 dim_in과 해상도에 맞도록 순차 다운/채널 정렬
            for sampler in self.control_cond_downsamplers:
                h_c = sampler(h_c)                  # (B, 64/128/256/..., H/2^k, W/2^k)
                cond_feats.append(h_c)              # stage 1..N
            cond_mid = h_c                           # mid 해상도 cond (dim = mid_dim)
            #---------------------------------------2025.10.11 ControlNet 도입부 수정

        h = []

        # 1) DOWN
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            # locked path: 입력 x로 그대로 진행
            # trainable copy path: (선택) pre-zero(conv(cond))를 입력에만 더해 별도 경로로 처리
            if self.use_control and self.control_inject_down and (cond is not None):
                x_ctrl_in = x + self.ctrl_down_pre[i](cond_feats[i])
                c1, c2, cattn = self.ctrl_down_copy[i]
                x_ctrl = c1(x_ctrl_in, t)
                x_ctrl = c2(x_ctrl, t)
                x_ctrl = cattn(x_ctrl) + x_ctrl
                # post zero-conv로 정렬 후 locked 출력(y)에 잔차로 합산 (웨이트X, 활성값O)
                ctrl_residual = self.ctrl_down_post[i](x_ctrl)
            else:
                ctrl_residual = None

            # locked 경로 실제 계산
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x

            # trainable copy의 post-zero 출력을 여기서 합산
            if ctrl_residual is not None:
                x = x + ctrl_residual

            h.append(x)
            x = downsample(x)

        # 2) MID
        if self.use_control and self.control_inject_mid and (cond is not None):
            x_ctrl_mid_in = x + self.ctrl_mid_pre(cond_mid)     # pre zero-conv(cond -> mid_dim), trainable 입력에만
            m1, mattn, m2 = self.ctrl_mid_copy
            x_ctrl_mid = m1(x_ctrl_mid_in, t)
            x_ctrl_mid = mattn(x_ctrl_mid) + x_ctrl_mid
            x_ctrl_mid = m2(x_ctrl_mid, t)
            ctrl_mid_residual = self.ctrl_mid_post(x_ctrl_mid)  # post zero-conv
        else:
            ctrl_mid_residual = None

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        if ctrl_mid_residual is not None:
            x = x + ctrl_mid_residual

        # (옵션) cond 업해상도 준비
        if self.use_control and self.control_inject_up and (cond is not None):
            h_c = cond_mid
            cond_up_feats = []
            for upsampler in self.control_cond_upsamplers:
                cond_up_feats.append(h_c)
                h_c = upsampler(h_c)

        # 3) UP
        for i, (block1, block2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim = 1)

            if self.use_control and self.control_inject_up and (cond is not None):
                x_ctrl_in = x + self.ctrl_up_pre[i](cond_up_feats[i])
                u1, u2, uattn = self.ctrl_up_copy[i]
                x_ctrl = u1(x_ctrl_in, t)
                # up의 두 번째 블록은 concat 후에 들어가므로 잠시 뒤에 copy도 동일한 순서를 맞춘다
                x = block1(x, t)

                # concat 두 번째 skip
                x = torch.cat((x, h.pop()), dim = 1)
                # trainable copy도 동일하게 두 번째 입력을 구성해야 하지만,
                # 간단히는 u2를 x_ctrl에 바로 적용(표현력은 충분)
                x_ctrl = u2(x_ctrl, t)
                x_ctrl = uattn(x_ctrl) + x_ctrl
                ctrl_up_residual = self.ctrl_up_post[i](x_ctrl)

                x = block2(x, t)
                x = attn(x) + x
                x = x + ctrl_up_residual
            else:
                x = block1(x, t)
                x = torch.cat((x, h.pop()), dim = 1)
                x = block2(x, t)
                x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)



# gaussian diffusion trainer class

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) # (b, 1, 1, 1) 배치마다 다른 스칼라가 전체 공간(C,H,W)에 적용

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1 # 
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps # 0~1 균등분포
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2  # 0~1 코사인 곡선
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # 1로 정규화
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # 1 - (α_t / α_(t-1)) = β_t
    return torch.clip(betas, 0, 0.999)  # 0~0.999 사이로 클램핑

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid() # start =-3, tau=1 → 0.0474
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False,
        # ▼ 새로 추가: MIP 손실 관련
        mip_options=None,     # <- MIPloss가 필요로 하는 설정만 담은 객체/딕트
        lambda_mip=1.0,       # <- MIP 손실 가중치
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels # 3
        self.self_condition = self.model.self_condition # False

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'   # (H, W) 
        self.image_size = image_size    # (H, W) 320, 320

        self.objective = objective # 'pred_v'

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)



        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps # True
        self.is_ddim_sampling = self.sampling_timesteps < timesteps # True
        self.ddim_sampling_eta = ddim_sampling_eta  # 0.0

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # 

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # mip loss 2025-10-07
        self.lambda_mip = float(lambda_mip)
        self.mip_loss = None   # 일단 None으로 두고, 버퍼 등록 후 디바이스가 정해지
        self.awl = None        # AWL도 기본 None으로
        # (중략) betas, alphas 등 register_buffer 끝난 뒤에 생성해야 self.device 사용 가능
        if mip_options is not None:
            self._init_mip_loss(mip_options)


        # immiscible diffusion

        self.immiscible = immiscible

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def _init_mip_loss(self, mip_options):
        """
        MIP 손실, AWL, 시그마 스케줄 초기화
        - dict/객체 모두 지원
        - num_slice는 채널 수와 동기화
        """
        def _get(obj, path, default=None):
            cur = obj
            for key in path.split('.'):
                if isinstance(cur, dict):
                    cur = cur.get(key, None)
                else:
                    cur = getattr(cur, key, None)
                if cur is None:
                    return default
            return cur

        temp = _get(mip_options, 'mip_loss.temp', 1.0)
        num_slice = _get(mip_options, 'data.use_slice', self.channels)

        # MIPloss가 속성 접근을 기대하므로 얕은 네임스페이스 구성
        class _Opt: pass
        opt = _Opt()
        opt.mip_loss = _Opt()
        opt.mip_loss.temp = float(temp)
        opt.data = _Opt()
        opt.data.use_slice = int(num_slice if num_slice is not None else self.channels)

        self.mip_loss = MIPloss(opt).to(self.device)
        # 채널 수와 강제 동기화 (불일치 시 슬라이싱 이상 방지)
        if getattr(self.mip_loss, 'num_slice', None) != self.channels:
            self.mip_loss.num_slice = self.channels

        # 결합 손실 2개(base, mip)에 맞춰 파라미터 개수 지정
        self.awl = AutomaticWeightedLoss(num=4).to(self.device)

        # timestep별 sigma 스케줄(여기서는 0으로 초기화 – 필요 시 바꾸면 됨)
        # sigma_sched = torch.zeros(self.num_timesteps, device=self.device, dtype=torch.float32) # sigma_sched = 항상 0으로 되어있는 상태임 2025-10-09
        # self.register_buffer('some_sigma_sched', sigma_sched)
        T = self.num_timesteps # 1000
        t = torch.arange(T, device=self.device, dtype=torch.float32)
        m = t / (T - 1)
        delta_t = 2.0 * (m - m * m)          # 논문의 δ_t
        self.register_buffer('some_sigma_sched', delta_t)

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise): # x_t: noisy image, t: time step, noise: predicted noise/ 노이즈가 섞인 상태 x_t와 해당 시점의 노이즈 ε가 있으면, 깨끗한 원본 x₀를 복원.
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0): # x_t: noisy image, t: time step, x0: predicted original image/ 노이즈가 섞인 상태 x_t와 해당 시점의 원본 이미지 x₀가 있으면, 그 시점의 노이즈 ε를 예측.
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v): # v에서 다시 x₀를 복원하는 공식입니다. (pred_v일 때 모델 출력 → x₀로 바꿔 쓰려면 필요)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t): # DDPM 샘플러라면 q_posterior가 매 스텝 필요
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # -------------------------------------------------20251010 수정 controlnet update (cond 전달 경로 추가)
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, *, cond=None):
        model_output = self.model(x, t, x_self_cond, cond=cond)  # ← cond 전달
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    # -------------------------------------------------20251010 수정 controlnet update (cond 전달 경로 추가)
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True, *, cond=None): # 현재시점 X_t에서 이전시점 X_{t-1}로 넘어가기 위한 평균과 분산 즉 샘플링할 가우시안 파라미터를 만들어 줍니다.
        preds = self.model_predictions(x, t, x_self_cond, clip_x_start=False, rederive_pred_noise=False, cond=cond) # ModelPrediction(pred_noise, pred_x_start)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # -------------------------------------------------20251010 수정 controlnet update (cond 전달 경로 추가)
    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None, *, cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True, cond=cond)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    # -------------------------------------------------20251010 수정 controlnet update (cond 전달 경로 추가)
    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False, *, cond=None):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, cond=cond) # pred_img, pred_x_start
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1) # return_all_timesteps이 True면 (B, T, C, H, W) 텐서로 반환 / return_all_timesteps이 False면 (B, C, H, W) 텐서로 반환

        ret = self.unnormalize(ret)
        return ret

    # -------------------------------------------------20251010 수정 controlnet update (cond 전달 경로 추가)
    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False, *, cond=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, cond=cond)

            if time_next < 0: # time_next == -1
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    # -------------------------------------------------20251010 수정 controlnet update (cond 전달 경로 추가)
    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False, *, cond=None):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps, cond=cond)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img


    def noise_assignment(self, x_start, noise): # 이미지에 가장 잘 맞는(가까운) 노이즈”를 겹치지 않게 배분하는 함수 # i번째 x_start에는 어떤 noise를 붙이면 전체 거리 합이 최소가 되는가?”를 **헝가리안 알고리즘**(최소 비용 매칭)으로 찾아 인덱스 매핑을 돌
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise)) # (B, C, H, W) → (B, C*H*W) 배치 차원 b만 남기고 나머지 축을 전부 펼쳐서 2D로 만듭니다.
        dist = torch.cdist(x_start, noise) # (B, B) 각 배치 쌍마다의 거리/ 행렬 유클리드 거리(기본값)로 모든 쌍의 거리 행렬을 만듭니다. (행 = 각 x_start 샘플, 열 = 각 noise 샘플) 즉 dist[i, j]는 x_start[i]와 noise[j]의 거리.
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)  
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
    # -------------------------------------------------20251010 수정 controlnet update (cond 인자 추가: 학습 시 조건 전달)
    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None, *, return_stats: bool = False, cond=None): # MIP loss도 tqdm pbar로 나타내려고 추가한거임
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        # 이미지 전체(공간 전체)에 같은 오프셋을 넣어 채널 바이어스를 랜덤하게 주면 분포 커버리지가 넓어지면서 훈련 안정/품질이 좋아지는 보고가 있음
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength) # 0.0
        
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise) # noisy image / 정방향 샘플링으로 노이즈가 섞인 이미지 x_t를 만듭니다.

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                # -------------------------------------------------20251010 수정 controlnet update (cond 전달)
                x_self_cond = self.model_predictions(x, t, cond=cond).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        # -------------------------------------------------20251010 수정 controlnet update (cond 전달)
        model_out = self.model(x, t, x_self_cond, cond=cond)  # 구현 설정에 따라 ε / x₀ / v 중 하나를 직접 예측하도록 설계, 이 함수에선 “모델이 낸 것”을 model_out이라 두고, 그에 맞는 정답(target) 을 아래에서 만든다

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # ################################################################################## 
        # loss = F.mse_loss(model_out, target, reduction = 'none')
        # loss = reduce(loss, 'b ... -> b', 'mean')

        # loss = loss * extract(self.loss_weight, t, loss.shape) # self.loss_weight = SNR 기반 손실 가중치 (배치마다 다른 스칼라) 적용 # (B,) → (B, 1, 1, 1)로 변환되어 각 배치에 곱해짐 
        # return loss.mean()
        # ################################################################################## 원래 여기까지코드가 맞음 아래는 추가함
        
        # ===================== 👇👇👇 새 옵션 추가 👇👇👇 ===================== 
        ### SeoSY 2025-07-24 (기존 손실함수에서 MIP loss로 변경)
        base_loss = F.mse_loss(model_out, target, reduction = 'none')
        base_loss = reduce(base_loss, 'b ... -> b', 'mean')
        base_loss = base_loss * extract(self.loss_weight, t, base_loss.shape)
        base_loss = base_loss.mean()

        # MIP 비활성 시에는 기본 손실만 반환
        # ---------------------------------------------------------------------------2025-10-09 MIP loss를 pbar로 나타내기 Seo 👇
        # if getattr(self, 'mip_loss', None) is None or getattr(self, 'awl', None) is None:
        #     return base_loss
        # ----- base_loss만 쓰는 경로 ----- 원래 위에 2줄이였는데 MIP loss 도 pbar로 나타내려고 아래 줄로 바뀜
        if getattr(self, 'mip_loss', None) is None or getattr(self, 'awl', None) is None:
            if return_stats:
                stats = {'base': float(base_loss.detach().item()), 'mip': None, 'total': float(base_loss.detach().item())}
                return base_loss, stats
            return base_loss
        # ---------------------------------------------------------------------------------------2025-10-09 MIP loss를 pbar로 나타내기 Seo 👆

        ## pred_noise, pred_x0, pred_v 공통: 모델 출력 → x0_hat 구하기, no grad를 통해서 grad 연결 차단
        if self.objective in ('pred_noise', 'pred_v'):
            with torch.no_grad():
                a_bar_sqrt     = extract(self.sqrt_alphas_cumprod, t, x.shape)
                one_minus_sqrt = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        if self.objective == 'pred_noise':
            x0_hat = (x - one_minus_sqrt * model_out) / a_bar_sqrt
        elif self.objective == 'pred_x0':
            x0_hat = model_out                  # grad 연결 유지
        elif self.objective == 'pred_v':
            x0_hat = a_bar_sqrt * x - one_minus_sqrt * model_out
        ## 위 코드는 MIP loss용 x0_hat 구하는 코드 x0hat은 (B, C, H, W) 텐서이며 각 배치에 대한 모델의 x₀ 예측을 담고 있습니다.

        # 형태확인용 print(x0_hat.shape) # (B, C, H, W)
        # 필요하다면 view/permute로 슬라이스 축이 dim=1에 오도록 바꿔서 사용
        # [B, D, H, W] 라면 D가 채널이자 슬라이스 축
        assert x0_hat.dim() == 4 and x0_hat.size(0) == x_start.size(0) 

        # MIPloss의 slice 수를 채널에 맞춰 강제 동기화(옵션 불일치 대비)
        if getattr(self.mip_loss, 'num_slice', None) != x0_hat.shape[1]:
            self.mip_loss.num_slice = x0_hat.shape[1]

        # MIP loss
        mip_loss = self.mip_loss(x0_hat, x_start)

        # ===================== 여기부터 추가: 혈관가중 L2 =====================
        # GT TOF가 밝은 곳일수록 중요하게 보도록 weight map 생성
        with torch.no_grad():
            tau   = 0.15   # 밝기 기준 (데이터 보고 조절)
            sharp = 5.0   # 경사
            m = torch.sigmoid((x_start - tau) * sharp)    # (B, C, H, W), 혈관일수록 1에 가까움

        lambda_vessel = 3.0
        weight_map = 1.0 + lambda_vessel * m
        vessel_weighted_l2 = (weight_map * (x0_hat - x_start).pow(2)).mean()
        # ================================================================

        # ===================== Edge-aware 손실 추가 =====================
        def sobel_filter(x):
            # x: (B, C, H, W)  — C채널 각각에 동일 Sobel 적용
            B, C, H, W = x.shape

            kx = torch.tensor(
                [[[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]],
                dtype=x.dtype,
                device=x.device
            ).unsqueeze(0)          # [1,1,3,3]

            ky = torch.tensor(
                [[[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]]],
                dtype=x.dtype,
                device=x.device
            ).unsqueeze(0)          # [1,1,3,3]

            # 각 채널별로 같은 커널을 쓰기 위해 C개로 expand 후 groups=C 사용
            kx = kx.expand(C, 1, 3, 3)     # [C,1,3,3]
            ky = ky.expand(C, 1, 3, 3)     # [C,1,3,3]

            gx = F.conv2d(x, kx, padding=1, groups=C)   # (B,C,H,W)
            gy = F.conv2d(x, ky, padding=1, groups=C)   # (B,C,H,W)

            return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        with torch.no_grad():
            gt_edge = sobel_filter(x_start)
        pred_edge = sobel_filter(x0_hat)
        loss_edge = F.l1_loss(pred_edge, gt_edge)
        # ================================================================


        # 5) 결합
        # (a) 간단 가중합
        # lam_mip = getattr(self, 'lambda_mip', 1.0)   # 하이퍼파라미터
        # total_loss = base_loss + lam_mip * mip_loss

        # (b) AutomaticWeightedLoss로 불확실성 기반 가중(원하면)
        awl_losses = [base_loss, mip_loss]

        # ---------------------------------------------------------------------------2025-10-09 MIP loss를 배치마다 다른 delta t로 적용하기 mean 않하고 Seo 👇
        # sigma_t = extract(self.some_sigma_sched, t, x.shape).mean() # sigma_t는 텐서 스칼라로(브로드캐스트/shape 이슈 방지). timestep별 버퍼에서 추출 후 평균(이러면 배치 8만큼 ).
        # total_loss = self.awl(awl_losses, sigma_t=sigma_t)
        ### 그냥 아래 꺼 말고 위에 두줄쓰는게 속편함
        # 변경된 부분: 배치 평균이 아닌 샘플별 sigma_t (δ_t) 벡터 사용 ⬇ 👇  
        B = x.size(0)       # 샘플별 sigma (δ_t) 벡터 [B]
        sigma_vec = extract(self.some_sigma_sched, t, (B, 1, 1, 1)).squeeze(-1).squeeze(-1).squeeze(-1)  # [B]

        # AWL 파라미터로 조합을 호출부에서 직접 계산 (AWL 코드는 손대지 않음)
        theta0_sq = self.awl.params[0] ** 2   # base용
        theta1_sq = self.awl.params[1] ** 2   # mip용
        theta2_sq = self.awl.params[2] ** 2   # vessel-weighted용
        theta3_sq = self.awl.params[3] ** 2   # edge

        # base_loss, mip_loss는 현재 스칼라임(배치 평균). 
        # base는 스칼라 분모, MIP는 샘플별 분모를 사용 → per-sample 가중 근사
        adj_base = theta0_sq                           # scalar
        adj_mip  = theta1_sq + sigma_vec               # [B]  ← 원래 네가 per-sample로 하던 거 유지
        adj_vw   = theta2_sq                           # scalar (필요하면 여기도 per-sample로 바꿀 수 있음)

        # ----- 여기부터 수정: edge 쪽에 t-기반 sigmoid 스케줄 반영 -----
        # t를 0~1로 정규화해서 초기에는 edge 약하게, 후반에는 강하게
        T = float(self.num_timesteps)                  # 예: 1000
        t_norm = t.float().view(-1) / (T - 1.0)        # [B], 0~1 범위

        k      = 8.0   # sigmoid 기울기 (클수록 후반에 급격히 켜짐)
        center = 0.5   # 중간쯤에서 전환
        edge_phase = torch.sigmoid((t_norm - center) * k)   # 초반≈0, 후반≈1  → [B]

        alpha_edge = 1.0
        # early: edge_phase≈0 → (1 - edge_phase)≈1 → adj_edge 커짐 → edge term 약하게
        # late:  edge_phase≈1 → (1 - edge_phase)≈0 → adj_edge≈theta3_sq → edge term 강하게
        adj_edge = theta3_sq + alpha_edge * (1.0 - edge_phase)   # [B]
        # ------------------------------------------------------

        total_per = (
            (0.5 / adj_base) * base_loss + torch.log(1 + adj_base)
            + (0.5 / adj_mip)  * mip_loss          + torch.log(1 + adj_mip)
            + (0.5 / adj_vw)   * vessel_weighted_l2 + torch.log(1 + adj_vw)
            + (0.5 / adj_edge) * loss_edge         + torch.log(1 + adj_edge)
        )

        total_loss = total_per.mean()
        # ---------------------------------------------------------------------------2025-10-09 MIP loss를 배치마다 다른 delta t로 적용하기 mean 않하고 Seo 👆

        if return_stats:
            stats = {
                'base':        float(base_loss.detach().item()),
                'mip':         float(mip_loss.detach().item()),
                'vessel_l2':   float(vessel_weighted_l2.detach().item()),
                'edge':        float(loss_edge.detach().item()),
                'total':       float(total_loss.detach().item()),
            }
            return total_loss, stats

        return total_loss
        # ---------------------------------------------------------------------------2025-10-09 MIP loss를 pbar로 나타내기 Seo 👆
        # ===================== 👆👆👆 새 옵션 추가 👆👆👆 =====================


    # -------------------------------------------------20251010 수정 controlnet update (cond 전달 경로 추가)
    def forward(self, img, *args, cond=None, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, cond=cond, **kwargs)





def gumbel_softmax_sample(logits, temperature, gumbel, dim):
    '''mip loss'''
    y = logits + gumbel
    return F.softmax(y / temperature, dim)


def cal_snr(noise_img, clean_img):
    noise_img, clean_img = noise_img.detach().cpu().numpy(), clean_img.detach().cpu().numpy()
    noise_signal = noise_img - clean_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal**2
    clean_signal_2 = clean_signal**2
    sum1 = np.sum(clean_signal_2)
    sum2 = np.sum(noise_signal_2)
    snrr = 20*math.log10(math.sqrt(sum1)/math.sqrt(sum2))
    return snrr


class MIPloss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.temp = options.mip_loss.temp
        self.num_slice = options.data.use_slice
        # self.L1 = torch.nn.L1Loss()
        self.L2 = torch.nn.MSELoss()

    def reset_gumbel(self, img_fake):
        U = torch.rand_like(img_fake)
        # U = torch.rand(img_fake.size()).cuda()
        self.gumbel = -torch.log(-torch.log(U + 1e-20) + 1e-20)  # sample_gumbel

    def forward(self, img_fake, target):
        self.reset_gumbel(img_fake)
        pred_mips_c1 = torch.zeros_like(img_fake)
        target_mips_c1 = torch.zeros_like(target)
        for idx in range(img_fake.shape[1]):
            pred_mip = gumbel_softmax_sample(img_fake[:, :idx+1], self.temp, self.gumbel[:, :idx+1], dim=1)
            target_mips_c1[:, idx] = torch.max(target[:, :idx+1], dim=1)[0]
            pred_mips_c1[:, idx] = torch.sum(pred_mip*img_fake[:, :idx+1], dim=1)

        pred_mips_c2 = torch.zeros_like(img_fake)
        target_mips_c2 = torch.zeros_like(target)
        for idx in range(img_fake.shape[1]):
            pred_mip = gumbel_softmax_sample(img_fake[:, self.num_slice-idx-1:], self.temp, self.gumbel[:, self.num_slice-idx-1:], dim=1)
            target_mips_c2[:, idx] = torch.max(target[:, self.num_slice-idx-1:], dim=1)[0]
            pred_mips_c2[:, idx] = torch.sum(pred_mip*img_fake[:, self.num_slice-idx-1:], dim=1)
    
        loss_ = self.L2(img_fake, target)
        loss_mip_c1 = self.L2(pred_mips_c1, target_mips_c1)
        loss_mip_c2 = self.L2(pred_mips_c2, target_mips_c2)
        loss = loss_ + loss_mip_c1 + loss_mip_c2

        return loss
        

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=4):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, losses, sigma_t):
        loss_sum = 0
       
        for i, loss in enumerate(losses):
            if i != 0:
                adjust_para = self.params[i] ** 2 + sigma_t # sigma_t = extract(self.some_sigma_sched, t, x.shape).mean()
            else:
                adjust_para = self.params[i] ** 2
            loss_sum += 0.5 / adjust_para * loss + torch.log(1 + adjust_para)

        return loss_sum





# # dataset classes

# class Dataset(Dataset):
#     def __init__(
#         self,
#         folder,
#         image_size,
#         exts = ['jpg', 'jpeg', 'png', 'tiff', 'npy'],
#         augment_horizontal_flip = False,
#         convert_image_to = None
#     ):
#         super().__init__()
#         self.folder = folder
#         self.image_size = image_size
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

#         maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

#         self.transform = T.Compose([
#             T.Lambda(maybe_convert_fn),
#             # T.Resize(image_size), # ★ Resize 제거, 센터크롭만
#             T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
#             T.CenterCrop(image_size),
#             T.ToTensor()
#         ])

#         # ✅ npy 전용 transform도 Compose로 구성
#         self.npy_transform = T.Compose([                                          ## SeoSY 2025-07-24 (기존에는 PIL에 대해서 transform이 작성되어있어서 추후에 npy도 업데이트 가능하도록 만들어뒀음)
#             T.Lambda(lambda x: x)  # identity                                     ## SeoSY 2025-07-24
#         ])                                                                        ## SeoSY 2025-07-24

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, index):
#         path = self.paths[index]
#         if path.suffix.lower() == '.npy':                                          ## SeoSY 2025-07-24 (npy 데이터도불러 올 수 있도록 추가함.)
#             arr = np.load(path)  # shape: (1, H, W), already [0,1]                 ## SeoSY 2025-07-24
#             img = torch.from_numpy(arr).float()                                    ## SeoSY 2025-07-24
#             return self.npy_transform(img)                                         ## SeoSY 2025-07-24
#         else:
#             img = Image.open(path)
#             return self.transform(img)


# #---------------------------------------------------------2025.10.11 dataloader관련 수정
# from torch.utils.data import Dataset as _TorchDataset

# class _PairedDataset(_TorchDataset):
#     """두 개의 H5WindowDataset을 같은 인덱스로 읽어 (a,b) 튜플을 반환"""
#     def __init__(self, ds_a, ds_b, strict: bool = True):
#         self.ds_a = ds_a
#         self.ds_b = ds_b
#         if strict and (len(ds_a) != len(ds_b)):
#             raise ValueError(f"paired dataset length mismatch: {len(ds_a)} vs {len(ds_b)}")
#         self.n = min(len(ds_a), len(ds_b)) if not strict else len(ds_a)
#     def __len__(self):
#         return self.n
#     def __getitem__(self, idx):
#         a = self.ds_a[idx]
#         b = self.ds_b[idx]
#         if isinstance(a, tuple): a = a[0]
#         if isinstance(b, tuple): b = b[0]
#         return (a, b)
# #---------------------------------------------------------2025.10.11 dataloader관련 수정

# trainer class
class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        # ==== 아래 5개가 추가된 인자 (HDF5용) ====
        # ==== HDF5용 데이터로더 인자 ====
        dataset_key = 'data',
        neighbors = 0,
        val_ratio = 0.05,
        seed = 42,
        rescale = 'none',
        # ▼ 추가 (크롭 + 검증 분할 제어)
        crop_size = None,                   # int or (h, w)
        crop_mode_train = 'random',         # 'random' | 'center'
        crop_mode_val   = 'center',         # 'center' 권장
        val_subject_count = None,           # 예: 10 (비율 대신 개수)
        val_subject_list  = None,           # ['subject_a', 'subject_b', ...]
        # ===================== 👇👇👇 새 옵션 추가 👇👇👇 =====================
        use_val_loss_for_best = False,     # True면 val loss로 best 저장을 가능하게
        val_subset_batches = 50,           # 평가에 사용할 미니배치 수(빠른 평가)
        eval_use_val_split = True,         # H5WindowDataset 'val' split 사용 시도
        # ===================== 👆👆👆 새 옵션 추가 👆👆👆 =====================

        # --------- 20251010 ControlNet paired loading: cond H5 전달용 인자 추가 ---------
        cond_h5_path: str | None = None,          # mGRE H5 경로 (ControlNet일 때만 사용)
        cond_dataset_key: str | None = None,      # mGRE H5 내 dataset key (기본: dataset_key)
        cond_rescale: str | None = None,           # mGRE rescale 모드 (기본: rescale)
        # --------------------------------------------------------------------------------
        best_policy: str = "val_loss",   # ← ADD

        #----------수정 2025-10-20: test 분할 관련 옵션 추가
        test_ratio: float = 0.10,
        test_subject_count = None,
        test_subject_list = None,
        eval_use_test_split: bool = True,
        #----------수정 2025-10-20 끝

        # ### ---------------------------------수정: z-slice 범위 옵션 추가
        z_start: int | None = 30,
        z_end:   int | None = 80,
        # ### ---------------------------------수정 끝
    ):

        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # ===================== ✅ 멤버 선초기화 (AttributeError 방지) =====================
        self._is_paired = False  # cond_h5_path가 주어져 페어 로딩 경로를 탈 때 True
        self.ds = None
        self.ds_cond = None
        self.val_ds = None
        self.val_ds_cond = None
        self.val_dl = None
        self.best_val_loss = float('inf')
        #----------수정 2025-10-20: test 관련 멤버 초기화
        self.test_ds = None
        self.test_ds_cond = None
        self.test_dl = None
        #----------수정 2025-10-20 끝
        # ================================================================================

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels
        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.save_best_and_latest_only = save_best_and_latest_only          # [ADD] (이미 있다면 생략 가능)

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        img_size_hw = self.image_size
        if isinstance(img_size_hw, int):
            img_size_hw = (img_size_hw, img_size_hw)

        # ### ---------------------------------수정: z_start / z_end 멤버로 저장
        self.z_start = z_start
        self.z_end   = z_end
        # ### ---------------------------------수정 끝

        # --------- ControlNet paired loading (single .h5, tuple output) ---------
        if str(folder).lower().endswith('.h5') and (cond_h5_path is not None):
            self._is_paired = True

            # 같은 파일이면 cond_h5_path를 folder로 정규화 (굳이 달라도 되지만 같으면 I/O 1회)
            if str(cond_h5_path) != str(folder):
                # 서로 다른 파일도 지원은 되지만, 한 파일에서 키만 다르게 쓰려는 목적이면 통일
                cond_h5_path = folder

            paired_train_ds = H5WindowDataset_CLDM(
                h5_path          = folder,           # 한 번만 넘김
                target_key       = 'tof',
                cond_key         = 'mgre',
                image_size       = img_size_hw,      # (H,W) 체크용
                neighbors        = neighbors,
                split            = 'train',
                val_ratio        = 0.1,
                val_subject_count= val_subject_count,
                val_subject_list = val_subject_list,
                seed             = seed,
                rescale          = rescale,                 # target 스케일
                cond_rescale     = (cond_rescale or rescale),  # cond 스케일
                horizontal_flip  = True,              # 정합 유지안하고 훈련시에는 flip 허용
                crop_size        = crop_size,
                crop_mode_train  = 'center',           # 랜덤 크롭 금지(정합)
                crop_mode_val    = crop_mode_val,
                return_meta      = False,              # 필요 시 True
                #----------수정 2025-10-20: test 인자 전달
                test_ratio       = test_ratio,
                test_subject_count = test_subject_count,
                test_subject_list  = test_subject_list,
                #----------수정 2025-10-20 끝

                # ### ---------------------------------수정: z 슬라이스 제한 전달
                z_start          = self.z_start,
                z_end            = self.z_end,
                # ### ---------------------------------수정 끝
            )

            dl = DataLoader(
                paired_train_ds,
                batch_size = train_batch_size,
                shuffle = True,
                pin_memory = True,
                num_workers = 8, # max(1, cpu_count() // 2),                         # HDF5 락 방지: 워커 축소
                persistent_workers = True,
            )
            dl = self.accelerator.prepare(dl)
            self.dl = cycle(dl)
            #---------------------------------------------------------2025.10.11 dataloader관련 수정

        else:
            if str(folder).lower().endswith('.h5'):
                # ✅ HDF5 동적 윈도우 데이터셋
                self.ds = H5WindowDataset(
                    h5_path = folder,
                    dataset_key = dataset_key,
                    image_size = img_size_hw,
                    neighbors = neighbors,
                    split = 'train',
                    # ----- 검증 분할 -----
                    val_ratio = val_ratio,
                    val_subject_count = val_subject_count,
                    val_subject_list = val_subject_list,
                    seed = seed,
                    # ----- 정규화 & 증강 -----
                    rescale = rescale,
                    horizontal_flip = augment_horizontal_flip,
                    # ----- 크롭 -----
                    crop_size = crop_size,
                    crop_mode_train = crop_mode_train,
                    crop_mode_val   = crop_mode_val,
                    #----------수정 2025-10-20: test 인자 전달
                    test_ratio = test_ratio,
                    test_subject_count = test_subject_count,
                    test_subject_list  = test_subject_list,
                    #----------수정 2025-10-20 끝

                    # ### ---------------------------------수정: z 슬라이스 제한 전달
                    z_start = self.z_start,
                    z_end   = self.z_end,
                    # ### ---------------------------------수정 끝
                )
            else:
                # ✅ 기존 폴더 이미지 데이터셋
                self.ds = Dataset(
                    folder, self.image_size,
                    augment_horizontal_flip = augment_horizontal_flip,
                    convert_image_to = convert_image_to
                )
            # 🔎 여기서 바로 길이 출력
            try:
                _tqdm.write(f"[DATASET] len(self.ds) = {len(self.ds)}")
            except Exception as e:
                _tqdm.write(f"[DATASET] len(self.ds) 확인 중 오류: {e}")

            assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

            dl = DataLoader(
                self.ds,
                batch_size = train_batch_size,
                shuffle = True,
                pin_memory = True,
                num_workers = 8, # cpu_count()
            )

            # 🔎 DataLoader 정보도 참고로 출력
            try:
                _tqdm.write(f"[DATALOADER] batch_size = {train_batch_size}, num_workers = {cpu_count()}")
                _tqdm.write(f"[DATALOADER] dataset length seen by dl = {len(dl.dataset)}")
            except Exception:
                pass

            dl = self.accelerator.prepare(dl)
            self.dl = cycle(dl)

        # ===================== 👇👇👇 새 멤버 초기화 👇👇👇 =====================
        self.use_val_loss_for_best = use_val_loss_for_best
        self.val_subset_batches    = val_subset_batches
        self.eval_use_val_split    = eval_use_val_split
        #----------수정 2025-10-20: test 사용 여부 저장
        self.eval_use_test_split   = eval_use_test_split
        #----------수정 2025-10-20 끝

        self.val_dl = None
        self.best_val_loss = float('inf')
        # ===================== 👆👆👆 새 멤버 초기화 👆👆👆 =====================

        # ===================== 👇👇👇 best-policy 상태값 초기화 (ADD) 👇👇👇 =====================
        self.best_policy = (best_policy or "val_loss").lower()  # ← ADD
        self.best_scores = {                                     # ← ADD
            "FID": float("inf"),
            "val_loss": float("inf"),
            "base": float("inf"),
            "mip": float("inf"),
            "total": float("inf"),
            "total_simple": float("inf"),
            "combo": float("inf"),
        }
        # ===================== 👆👆👆 best-policy 상태값 초기화 (ADD) 👆👆👆 =====================


        # --------- ControlNet paired loading: 검증도 페어 경로 구성 ---------
        if self._is_paired and self.eval_use_val_split:
            # 같은 파일에서 키만 다르게 쓸 목적이면 cond_h5_path를 folder로 통일
            if str(cond_h5_path) != str(folder):
                cond_h5_path = folder

            try:
                paired_val_ds = H5WindowDataset_CLDM(
                    h5_path            = folder,          # 한 번만 넘김
                    target_key         = 'tof',
                    cond_key           = 'mgre',
                    image_size         = img_size_hw,
                    neighbors          = neighbors,
                    split              = 'val',
                    val_ratio          = val_ratio,
                    val_subject_count  = val_subject_count,
                    val_subject_list   = val_subject_list,
                    seed               = seed,
                    rescale            = rescale,                     # target 스케일
                    cond_rescale       = (cond_rescale or rescale),   # cond 스케일
                    horizontal_flip    = False,                       # 검증 aug 금지
                    crop_size          = crop_size,
                    crop_mode_train    = 'center',
                    crop_mode_val      = 'center',
                    return_meta        = False,
                    #----------수정 2025-10-20: test 인자 전달(검증에서도 동일 분할 유지 목적)
                    test_ratio         = test_ratio,
                    test_subject_count = test_subject_count,
                    test_subject_list  = test_subject_list,
                    #----------수정 2025-10-20 끝

                    # ### ---------------------------------수정: z 슬라이스 제한 전달 (val)
                    z_start            = self.z_start,
                    z_end              = self.z_end,
                    # ### ---------------------------------수정 끝
                )
                _tqdm.write(f"[VAL] len(paired_val_ds) = {len(paired_val_ds)}")

                self.val_dl = DataLoader(
                    paired_val_ds,
                    batch_size = 1,
                    shuffle = False,
                    pin_memory = True,
                    num_workers = 8,  # max(1, cpu_count() // 2),
                    persistent_workers = True,
                )
                self.val_dl = self.accelerator.prepare(self.val_dl)

            # except Exception as e:
            #     _tqdm.write(f"[VAL] building paired val loader failed; fallback to train stream for eval: {e}")
            #     self.val_dl = None
            
            except Exception as e:
                _tqdm.write(f"[VAL] building paired val loader failed: {e}")
                raise   # 평가 누수 방지를 위해 즉시 실패            #---------------------------------------------------------2025.10.11 dataloader관련 수정

        # ===================== ✅ 싱글 전용 val 분기: 방어적 체크로 수정 =====================
        if (not self._is_paired) and isinstance(getattr(self, "ds", None), H5WindowDataset) and self.eval_use_val_split:
            try:
                self.val_ds = H5WindowDataset(
                    h5_path = folder,
                    dataset_key = dataset_key,
                    image_size = img_size_hw,
                    neighbors = neighbors,
                    split = 'val',
                    val_ratio = val_ratio,
                    val_subject_count = val_subject_count,
                    val_subject_list = val_subject_list,
                    seed = seed,
                    rescale = rescale,
                    horizontal_flip = False,  # 검증에는 보통 augmentation 끔
                    crop_size = crop_size,
                    crop_mode_train = crop_mode_train,
                    crop_mode_val   = crop_mode_val,
                    # -------------------------------------------------20251010 수정 controlnet update: 메타와 함께 반환 (subject/z 추적)
                    return_meta = True,
                    #----------수정 2025-10-20: test 인자 전달
                    test_ratio = test_ratio,
                    test_subject_count = test_subject_count,
                    test_subject_list  = test_subject_list,
                    #----------수정 2025-10-20 끝

                    # ### ---------------------------------수정: z 슬라이스 제한 전달 (val)
                    z_start = self.z_start,
                    z_end   = self.z_end,
                    # ### ---------------------------------수정 끝
                )
                _tqdm.write(f"[VAL] len(self.val_ds) = {len(self.val_ds)}")
                val_dl = DataLoader(
                    self.val_ds,
                    # -------------------------------------------------20251010 수정 controlnet update: 볼륨 순서 보존을 위해 batch_size=1 고정
                    batch_size = 1,
                    # -------------------------------------------------20251010 수정 controlnet update: 검증에서는 셔플 금지
                    shuffle = False,
                    pin_memory = True,
                    num_workers = 8, # max(1, cpu_count() // 2)
                )
                self.val_dl = self.accelerator.prepare(val_dl)
            # except Exception as e:
            #     _tqdm.write(f"[VAL] building val loader failed; fallback to train stream for eval: {e}")
            #     self.val_dl = None
            except Exception as e:
                _tqdm.write(f"[VAL] building val loader failed: {e}")
                raise   # 평가 누수 방지를 위해 즉시 실패
        # ===================== 👆👆👆 싱글 전용 val 분기 끝 👆👆👆 =====================

        #----------수정 2025-10-20: TEST 분기 추가 (paired / single 모두)
        # 페어드 ControlNet 테스트 로더
        if self._is_paired and self.eval_use_test_split:
            if str(cond_h5_path) != str(folder):
                cond_h5_path = folder
            try:
                paired_test_ds = H5WindowDataset_CLDM(
                    h5_path            = folder,
                    target_key         = 'tof',
                    cond_key           = 'mgre',
                    image_size         = img_size_hw,
                    neighbors          = neighbors,
                    split              = 'test',
                    val_ratio          = val_ratio,
                    val_subject_count  = val_subject_count,
                    val_subject_list   = val_subject_list,
                    seed               = seed,
                    rescale            = rescale,
                    cond_rescale       = (cond_rescale or rescale),
                    horizontal_flip    = False,
                    crop_size          = crop_size,
                    crop_mode_train    = 'center',
                    crop_mode_val      = 'center',
                    return_meta        = False,
                    test_ratio         = test_ratio,
                    test_subject_count = test_subject_count,
                    test_subject_list  = test_subject_list,

                    # ### ---------------------------------수정: z 슬라이스 제한 전달 (test)
                    z_start            = self.z_start,
                    z_end              = self.z_end,
                    # ### ---------------------------------수정 끝
                )
                _tqdm.write(f"[TEST] len(paired_test_ds) = {len(paired_test_ds)}")
                self.test_dl = DataLoader(
                    paired_test_ds,
                    batch_size = 1,
                    shuffle = False,
                    pin_memory = True,
                    num_workers = 8,
                    persistent_workers = True,
                )
                self.test_dl = self.accelerator.prepare(self.test_dl)
            except Exception as e:
                _tqdm.write(f"[TEST] building paired test loader failed: {e}")
                raise

        # 싱글 테스트 로더
        if (not self._is_paired) and isinstance(getattr(self, "ds", None), H5WindowDataset) and self.eval_use_test_split:
            try:
                self.test_ds = H5WindowDataset(
                    h5_path = folder,
                    dataset_key = dataset_key,
                    image_size = img_size_hw,
                    neighbors = neighbors,
                    split = 'test',
                    val_ratio = val_ratio,
                    val_subject_count = val_subject_count,
                    val_subject_list = val_subject_list,
                    seed = seed,
                    rescale = rescale,
                    horizontal_flip = False,
                    crop_size = crop_size,
                    crop_mode_train = crop_mode_train,
                    crop_mode_val   = crop_mode_val,
                    return_meta = True,
                    test_ratio = test_ratio,
                    test_subject_count = test_subject_count,
                    test_subject_list  = test_subject_list,

                    # ### ---------------------------------수정: z 슬라이스 제한 전달 (test)
                    z_start = self.z_start,
                    z_end   = self.z_end,
                    # ### ---------------------------------수정 끝
                )
                _tqdm.write(f"[TEST] len(self.test_ds) = {len(self.test_ds)}")
                test_dl = DataLoader(
                    self.test_ds,
                    batch_size = 1,
                    shuffle = False,
                    pin_memory = True,
                    num_workers = 8,
                )
                self.test_dl = self.accelerator.prepare(test_dl)
            except Exception as e:
                _tqdm.write(f"[TEST] building test loader failed: {e}")
                raise
        #----------수정 2025-10-20 끝






        # optimizer
        # -------------------------------------------------20251010 수정 controlnet update (requires_grad=True 파라미터만 최적화)
        trainable_params = [p for p in diffusion_model.parameters() if p.requires_grad]
        self.opt = Adam(trainable_params, lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # (ADD) TensorBoard writer
        self.tb = None
        if self.accelerator.is_main_process:
            try:
                self.tb = SummaryWriter(log_dir=str(self.results_folder / "tb"))
            except Exception as e:
                _tqdm.write(f"[TB] SummaryWriter init failed: {e}")


        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        ## ==================== 👇👇👇 변경 시작 👇👇👇 =====================
        if save_best_and_latest_only:
            # FID 또는 val loss 중 하나는 켜져 있어야 함
            if not (calculate_fid or use_val_loss_for_best):
                raise AssertionError(
                    "`save_best_and_latest_only=True`이면 평가 기준이 필요합니다. "
                    "FID(calculate_fid=True) 또는 val loss(use_val_loss_for_best=True) 중 하나를 켜세요."
                )
            # 초기값들 준비 (둘 다 켠 경우 둘 다 초기화)
            if calculate_fid:
                self.best_fid = 1e10
            if use_val_loss_for_best:
                 self.best_val_loss = float('inf')
        ## ==================== 👆👆👆 변경 끝 👆👆👆 =====================


        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        # ## ==================== 👇👇👇 변경 시작 👇👇👇 ===================== 2025-10-07 SeoSY
        models_dir = self.results_folder / 'models' # models라는 폴더를 만들어서 깔끔하게 정리하려고 추가한 것임.
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(self.results_folder / f'models/model-{milestone}.pt')) # models에 들어갈 수 있도로 경로 설정
        ## ==================== 👆👆👆 변경 끝 👆👆👆 =====================

    # ---- (ADD) 작은 유틸: 원자적 텍스트 쓰기 ----
    @staticmethod
    def _atomic_write_text(path: Path, text: str):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)

    # ---- (ADD) best/latest 메타 기록 ----
    def _record_ckpt_info(self, kind: str, *, step: int, epoch: float | None = None,
                          metric_name: str | None = None, metric_value: float | None = None,
                          extra: dict | None = None):
        """
        kind: 'best' or 'latest'
        results_folder/models/model-{kind}.info.json 에 초소형 메타 기록
        """
        import json, time
        models_dir = self.results_folder / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "epoch": (None if epoch is None else float(epoch)),
            "metric_name": metric_name,
            "metric_value": (None if metric_value is None else float(metric_value)),
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            payload.update(extra)
        info_path = models_dir / f"model-{kind}.info.json"
        self._atomic_write_text(info_path, json.dumps(payload, ensure_ascii=False))


    # ==================== 👇👇👇 수정: 유연한 체크포인트 로딩 👇👇👇 ====================
    def load(self, milestone, checkpoint_dir=None, strict=True, load_optimizer=True, load_ema=True):
        accelerator = self.accelerator
        device = accelerator.device

        # ---- (수정) 외부 디렉토리에서도 로드 가능 ----
        base_dir = Path(checkpoint_dir) if checkpoint_dir is not None else self.results_folder
        ckpt_path = base_dir / f'models/model-{milestone}.pt'

        data = torch.load(str(ckpt_path), map_location=device, weights_only=True)

        # ---- 모델 가중치: strict 인자로 유연 로딩 ----
        model = self.accelerator.unwrap_model(self.model)
        missing, unexpected = model.load_state_dict(data['model'], strict=strict)
        if not strict:
            if missing:
                print(f"[load] (non-strict) missing keys: {len(missing)} (첫 5개) -> {missing[:5]}")
            if unexpected:
                print(f"[load] (non-strict) unexpected keys: {len(unexpected)} (첫 5개) -> {unexpected[:5]}")

        # ---- 옵티마이저 상태: 구조 달라지면 스킵 ----
        step_from_ckpt = int(data.get('step', 0))
        opt_loaded = False
        if load_optimizer and ('opt' in data) and (data['opt'] is not None):
            try:
                self.opt.load_state_dict(data['opt'])
                opt_loaded = True
            except Exception as e:
                print(f"[load] optimizer state skipped (reason: {e})")

        # ---- EMA 상태: 가능할 때만 로드, 실패 시 스킵 ----
        if load_ema and self.accelerator.is_main_process and ('ema' in data) and (data['ema'] is not None):
            try:
                self.ema.load_state_dict(data["ema"])
            except Exception as e:
                print(f"[load] EMA state skipped (reason: {e})")

        # ---- 스텝 결정: 옵티마이저 못 불렀으면 의미가 다르니 0으로 리셋 권장 ----
        if opt_loaded:
            self.step = step_from_ckpt
        else:
            print("[load] optimizer not loaded → step reset to 0 for safety")
            self.step = 0

        if 'version' in data:
            print(f"loading from version {data['version']}")

        # ---- GradScaler 상태도 가능하면 로드, 실패 시 스킵 ----
        try:
            if exists(self.accelerator.scaler) and exists(data.get('scaler', None)):
                self.accelerator.scaler.load_state_dict(data['scaler'])
        except Exception as e:
            print(f"[load] scaler state skipped (reason: {e})")
    # ==================== 👆👆👆 수정 끝 👆👆👆 ====================



    # (교체) _quick_eval_loss: float -> dict 리턴
    # 파일: denoising_diffusion_pytorch_SEO_cldm.py (혹은 Trainer 정의된 파일)
    @torch.inference_mode()
    def _quick_eval_loss(self, steps: int | None = None):
        """
        검증 서브셋에서 평균 loss를 계산해 dict로 반환하고, 텐서보드에 기록한다.
        반환 dict 키:
        - "val_total"          : 평균 total (기존 val_loss에 해당)
        - "val_base"           : 평균 base(MSE)
        - "val_mip"            : 평균 mip
        - "val_total_simple"   : 평균 (base + mip(있으면))
        - "val_combo"          : 평균 combo (모델이 제공할 때만)
        """

        was_training = self.model.training
        self.model.eval()

        device = self.device
        steps = steps or getattr(self, "val_subset_batches", 8)
        ema_model = self.ema.ema_model if hasattr(self, "ema") else self.model

        # ============== helpers ==============
        def _tofloat(x):
            if x is None: return None
            return float(x.item()) if hasattr(x, "item") else float(x)

        def _to_tensor_on_device(x):
            # dict 같은 건 텐서화하지 않음
            import torch, numpy as np
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device)
            try:
                return torch.as_tensor(x, device=device)
            except Exception:
                return None

        def _extract_tof_cond(batch):
            # dict: 이미지/조건 키만 뽑아서 텐서로, meta 등은 무시
            if isinstance(batch, dict):
                tof = None
                # 우선순위: 'tof' -> self.dataset_key -> 'data' -> 'img' -> 'image'
                for k in ("tof", getattr(self, "dataset_key", None), "data", "img", "image"):
                    if k and (k in batch):
                        tof = _to_tensor_on_device(batch[k])
                        if tof is not None:
                            break
                cond = None
                for k in ("mgre", "cond"):
                    if k in batch:
                        cond = _to_tensor_on_device(batch[k])
                        if cond is not None:
                            break
                return tof, cond

            # tuple/list: 텐서화 가능한 첫 요소 tof, 다음 요소 cond
            if isinstance(batch, (tuple, list)):
                tof, cond = None, None
                for elem in batch:
                    t = _to_tensor_on_device(elem)
                    if t is None:
                        continue
                    if tof is None:
                        tof = t
                    elif cond is None:
                        cond = t
                        break
                return tof, cond

            # 기타: 단일 텐서/배열로 간주
            return _to_tensor_on_device(batch), None
        # =====================================

        # val_dl 있으면 사용, 없으면 train stream에서 일부 스텝 샘플
        iterator = iter(self.val_dl) if getattr(self, "val_dl", None) is not None else None

        sums = dict(total=0.0, base=0.0, mip=0.0, combo=0.0)
        cnts = dict(total=0,   base=0,   mip=0,   combo=0)

        for _ in range(max(1, steps)):
            try:
                batch = next(iterator) if iterator is not None else next(self.dl)
            except StopIteration:
                break

            tof, cond = _extract_tof_cond(batch)
            if tof is None:
                # 이미지를 못 찾으면 해당 스텝은 스킵
                continue

            with self.accelerator.autocast():
                try:
                    out = ema_model(tof, cond=cond, return_stats=True)
                    if isinstance(out, (tuple, list)) and len(out) == 2:
                        loss, stats = out
                    else:
                        # (loss, stats) 형태가 아닐 수 있음
                        loss, stats = out, None
                except TypeError:
                    loss, stats = ema_model(tof, cond=cond), None

            # 구성 요소 추출
            tot   = _tofloat(stats.get("total", loss)) if isinstance(stats, dict) else _tofloat(loss)
            base  = _tofloat(stats.get("base"))  if isinstance(stats, dict) else None
            mip   = _tofloat(stats.get("mip"))   if isinstance(stats, dict) else None
            combo = _tofloat(stats.get("combo")) if isinstance(stats, dict) else None

            if tot  is not None: sums["total"] += tot;  cnts["total"] += 1
            if base is not None: sums["base"]  += base; cnts["base"]  += 1
            if mip  is not None: sums["mip"]   += mip;  cnts["mip"]   += 1
            if combo is not None:sums["combo"] += combo;cnts["combo"] += 1

        def _avg(name):
            return (sums[name] / max(1, cnts[name])) if cnts[name] > 0 else None

        avg_total = _avg("total")
        avg_base  = _avg("base")
        avg_mip   = _avg("mip")
        avg_combo = _avg("combo")
        avg_total_simple = (avg_base + (avg_mip or 0.0)) if (avg_base is not None) else None

        # 텐서보드 로깅 (여기서만 기록하면 중복 없음)
        if getattr(self, "tb", None) is not None:
            if avg_total is not None:
                self.tb.add_scalar("val/loss",  float(avg_total),         global_step=self.step)  # 호환 alias
                self.tb.add_scalar("val/total", float(avg_total),         global_step=self.step)
            if avg_base is not None:
                self.tb.add_scalar("val/base",  float(avg_base),          global_step=self.step)
            if avg_mip is not None:
                self.tb.add_scalar("val/mip",   float(avg_mip),           global_step=self.step)
            if avg_total_simple is not None:
                self.tb.add_scalar("val/total_simple", float(avg_total_simple), global_step=self.step)
            if avg_combo is not None:
                self.tb.add_scalar("val/combo", float(avg_combo),         global_step=self.step)

        val_stats = {
            "val_total":         avg_total,
            "val_base":          avg_base,
            "val_mip":           avg_mip,
            "val_total_simple":  avg_total_simple,
            "val_combo":         avg_combo,
        }
        # 다른 곳에서 참조 가능하도록 저장
        self._last_val_stats = val_stats

        if was_training:
            self.model.train()

        return val_stats

    def _maybe_update_bests(self, fid_score, eval_loss, val_stats):
        """
        best_policy에 따라 best_* 체크포인트들을 갱신하고, info.json + TB를 기록.
        """
        # latest는 항상
        self.save("latest")
        try:
            self._record_ckpt_info("latest", step=self.step, epoch=getattr(self,"epoch",None),
                                metric_name=None, metric_value=None)
        except Exception as e:
            self.accelerator.print(f"[meta] latest info write skipped: {e}")

        # 후보 지표 구성
        candidates = {}
        if fid_score is not None:
            candidates["FID"] = float(fid_score)
        if eval_loss is not None:
            candidates["val_loss"] = float(eval_loss)
        if isinstance(val_stats, dict):
            for src, dst in [
                ("val_base", "base"),
                ("val_mip", "mip"),
                ("val_total", "total"),
                ("val_total_simple", "total_simple"),
                ("val_combo", "combo"),
            ]:
                v = val_stats.get(src, None)
                if v is not None:
                    candidates[dst] = float(v)

        # 정책 키 선택
        pol = (self.best_policy or "val_loss").lower()
        if pol == "mip":
            keys = ["mip"]
        elif pol == "base":
            keys = ["base"]
        elif pol == "total_simple":
            keys = ["total_simple"]
        elif pol == "val_loss":
            keys = ["val_loss"]
        elif pol == "any":
            keys = [k for k in ("base", "mip", "total_simple") if k in candidates]
            for extra in ("total", "combo"):
                if extra in candidates:
                    keys.append(extra)
        else:
            keys = ["val_loss"]

        improved = {}

        # FID 별도
        if "FID" in candidates and candidates["FID"] < self.best_scores["FID"]:
            self.best_scores["FID"] = candidates["FID"]
            self.save("best_FID")
            improved["FID"] = candidates["FID"]
            try:
                self._record_ckpt_info("best_FID", step=self.step, epoch=getattr(self,"epoch",None),
                                    metric_name="FID", metric_value=candidates["FID"],
                                    extra=(val_stats or {}))
            except Exception as e:
                self.accelerator.print(f"[meta] best_FID info write skipped: {e}")
            if getattr(self, "tb", None) is not None:
                self.tb.add_scalar("best/FID", float(candidates["FID"]), global_step=self.step)

        # 선택된 지표별 best 저장
        for k in keys:
            if k not in candidates:
                continue
            val = candidates[k]
            if val < self.best_scores.get(k, float("inf")):
                self.best_scores[k] = val
                milestone = f"best_{k}"
                self.save(milestone)
                improved[k] = val
                try:
                    self._record_ckpt_info(milestone, step=self.step, epoch=getattr(self,"epoch",None),
                                        metric_name=k, metric_value=val, extra=(val_stats or {}))
                except Exception as e:
                    self.accelerator.print(f"[meta] {milestone} info write skipped: {e}")
                if getattr(self, "tb", None) is not None:
                    self.tb.add_scalar(f"best/{k}", float(val), global_step=self.step)

        if improved and getattr(self, "tb", None) is not None:
            parts = [f"{kk}={improved[kk]:.6f}" for kk in sorted(improved.keys())]
            self.tb.add_text("best/info", f"best at step {self.step} — " + ", ".join(parts),
                            global_step=self.step)


    # ===================== visualization helpers (drop-in) =====================

    @staticmethod
    def _to01_from_input(x: torch.Tensor, mode: str = "minus1to1") -> torch.Tensor:
        """
        입력 배치 텐서를 [0,1] 범위로 변환.
        - mode == 'minus1to1': [-1,1] -> [0,1]
        - mode == 'zeroto1'  : [0,1]  -> [0,1] (그대로)
        """
        if mode == "zeroto1":
            y = x
        else:
            # 기본: [-1,1] 가정
            y = (x + 1) * 0.5
        return y.clamp(0, 1)   # <-- out-of-place

    @staticmethod
    def _to01_from_model(x: torch.Tensor) -> torch.Tensor:
        """
        모델 sample()이 self.unnormalize()를 거쳐 [0,1]로 나오는 전제.
        혹시 모를 수치 튀김을 방지하려 clamp만 수행.
        """
        return x.clamp(0, 1)   # <-- out-of-place

    @staticmethod
    def _percentile_stretch(x: torch.Tensor, q_low: float = 0.01, q_high: float = 0.99, eps: float = 1e-6) -> torch.Tensor:
        """
        퍼센타일 기반 대비 스트레치.
        배치/채널 별로 2D 공간(H,W) 축에 대해 q_low, q_high 퍼센타일을 구해 정규화.
        입력과 출력 모두 [0,1] 범위를 가정/유지.
        """
        assert x.dim() == 4, "expect [B,C,H,W]"
        ql = torch.quantile(x, q_low,  dim=(2, 3), keepdim=True)
        qh = torch.quantile(x, q_high, dim=(2, 3), keepdim=True)
        y = (x - ql) / (qh - ql + eps)
        return y.clamp(0, 1)   # <-- out-of-place

    @staticmethod
    def _maybe_to_rgb(x: torch.Tensor) -> torch.Tensor:
        """
        단일 채널이면 3채널로 복제(시각화 편의용).
        이미 3채널 이상이면 그대로 반환.
        """
        if x.dim() != 4:
            raise ValueError("expect [B,C,H,W]")
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    # ===================== end of visualization helpers =====================


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                #---------------------------------------2025.10.11 ControlNet 도입부 수정
                # 통계 수집 주기: Trainer에 stats_interval 속성이 있으면 사용, 없으면 기본 50
                stats_interval = getattr(self, "stats_interval", 50)
                collect_stats = (self.step % stats_interval == 0)
                # 이 스텝에서 로그에 쓸 마지막 loss 텐서(동기화는 통계 수집시에만)
                step_loss_tensor = None
                #---------------------------------------2025.10.11 ControlNet 도입부 수정 (끝)

                for _ in range(self.gradient_accumulate_every):
                    # -------------------------------------------------20251010 수정 controlnet update (배치 파싱 + cond 전달, 안전한 .to)
                    data = next(self.dl)
                    if isinstance(data, (tuple, list)) and len(data) == 2:
                        tof, mgre = data
                    elif isinstance(data, dict) and 'tof' in data and 'mgre' in data:
                        tof, mgre = data['tof'], data['mgre']
                    else:
                        tof, mgre = data, None

                    tof  = tof.to(device)
                    # mgre = mgre.to(device) if mgre is not None else None
                    mgre = (mgre.to(device) if (mgre is not None and torch.is_tensor(mgre)) else None)

                    with self.accelerator.autocast():
                        #---------------------------------------2025.10.11 ControlNet 도입부 수정
                        # N스텝마다만 통계 요청, 그 외에는 loss만 반환시켜 동기화 최소화
                        loss_out = self.model(tof, return_stats=collect_stats, cond=mgre)

                        if collect_stats:
                            # (loss, stats) 튜플
                            loss, stats = loss_out
                            last_stats = stats      # 누적 후 로그에 사용
                        else:
                            # 통계 미수집 시: loss 텐서만 받음
                            loss = loss_out
                            last_stats = None
                        #---------------------------------------2025.10.11 ControlNet 도입부 수정 (끝)

                        loss = loss / self.gradient_accumulate_every
                        # 여기서 .item() 하지 않음 → 동기화 방지
                        step_loss_tensor = loss.detach() if step_loss_tensor is None else (step_loss_tensor + loss.detach())

                    self.accelerator.backward(loss)
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                # # 진행바/로그는 main 프로세스에서만
                # if accelerator.is_main_process:
                #     # 1) 매 스텝 진행률 증가
                #     pbar.update(1)

                #     # 2) 통계 수집 스텝에서만 값 계산 + TB 로깅
                #     if collect_stats and step_loss_tensor is not None:
                #         def _to_float(x):
                #             if x is None:
                #                 return None
                #             return float(x.item()) if hasattr(x, "item") else float(x)

                #         total_loss = _to_float(step_loss_tensor)
                #         pbar.set_description(f"loss: {total_loss:.4f}")

                #         base_v = mip_v = tot_v = None
                #         if 'last_stats' in locals() and last_stats is not None:
                #             base_v = _to_float(last_stats.get("base"))
                #             mip_v  = _to_float(last_stats.get("mip"))
                #             tot_v  = _to_float(last_stats.get("total"))

                #             if mip_v is None:
                #                 pbar.set_postfix_str(f"base={base_v:.4f} total={tot_v:.4f}")
                #             else:
                #                 pbar.set_postfix_str(f"base={base_v:.4f} mip={mip_v:.4f} total={tot_v:.4f}")

                #         if self.tb is not None:
                #             self.tb.add_scalar("train/loss",  total_loss, global_step=self.step)
                #             if base_v is not None:
                #                 self.tb.add_scalar("train/base",  base_v,  global_step=self.step)
                #             if mip_v is not None:
                #                 self.tb.add_scalar("train/mip",   mip_v,   global_step=self.step)
                #             if tot_v is not None:
                #                 self.tb.add_scalar("train/total", tot_v,   global_step=self.step)

                #     # 3) 즉시 출력 반영(옵션)
                #     pbar.refresh()

                # 진행바/로그는 main 프로세스에서만
                if accelerator.is_main_process:
                    # 1) 매 스텝 진행률 증가
                    pbar.update(1)

                    # 2) 통계 수집 스텝에서만 값 계산 + TB 로깅
                    if collect_stats and step_loss_tensor is not None:
                        def _to_float(x):
                            if x is None:
                                return None
                            return float(x.item()) if hasattr(x, "item") else float(x)

                        total_loss = _to_float(step_loss_tensor)
                        pbar.set_description(f"loss: {total_loss:.4f}")

                        base_v = mip_v = tot_v = None
                        vwl2_v = edge_v = None   # 👈 추가

                        if 'last_stats' in locals() and last_stats is not None:
                            base_v   = _to_float(last_stats.get("base"))
                            mip_v    = _to_float(last_stats.get("mip"))
                            vwl2_v   = _to_float(last_stats.get("vessel_l2"))  # 👈 추가
                            edge_v   = _to_float(last_stats.get("edge"))       # 👈 추가
                            tot_v    = _to_float(last_stats.get("total"))

                            # 진행바 postfix 문자열 구성
                            if mip_v is None:
                                # MIP 비활성일 때
                                pbar.set_postfix_str(
                                    f"base={base_v:.4f} "
                                    f"vwl2={vwl2_v:.4f} "
                                    f"edge={edge_v:.4f} "
                                    f"total={tot_v:.4f}"
                                )
                            else:
                                # MIP 활성일 때
                                pbar.set_postfix_str(
                                    f"base={base_v:.4f} "
                                    f"mip={mip_v:.4f} "
                                    f"vwl2={vwl2_v:.4f} "
                                    f"edge={edge_v:.4f} "
                                    f"total={tot_v:.4f}"
                                )

                        if self.tb is not None:
                            self.tb.add_scalar("train/loss",  total_loss, global_step=self.step)
                            if base_v is not None:
                                self.tb.add_scalar("train/base",      base_v,  global_step=self.step)
                            if mip_v is not None:
                                self.tb.add_scalar("train/mip",       mip_v,   global_step=self.step)
                            if vwl2_v is not None:
                                self.tb.add_scalar("train/vessel_l2", vwl2_v,  global_step=self.step)  # 👈 추가
                            if edge_v is not None:
                                self.tb.add_scalar("train/edge",      edge_v,  global_step=self.step)  # 👈 추가
                            if tot_v is not None:
                                self.tb.add_scalar("train/total",     tot_v,   global_step=self.step)

                    # 3) 즉시 출력 반영(옵션)
                    pbar.refresh()


                # EMA는 모든 프로세스에서 업데이트
                if hasattr(self, "ema") and self.ema is not None:
                    self.ema.update()

                    
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()









                        # ===================== BEGIN PATCH (Seo 2025-09-30) =====================
                        # ---------- 저장/시각화 설정 ----------
                        n = int(math.sqrt(self.num_samples))
                        GRID_COLS = GRID_ROWS = n
                        GRID_COUNT = n * n

                        # vis 저장 모드: "center" | "all" | "tiles" | "center+all"
                        vis_save_mode = "center"

                        # 입력 텐서의 범위
                        VIS_INPUT_RANGE = 'minus1to1'

                        APPLY_STRETCH = False
                        Q_LOW, Q_HIGH = 0.01, 0.99
                        replicate_to_rgb = False

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every

                            # (A) 시각화용 배치
                            #--------고정그리드그림저장: train 로더 대신 val_dl의 앞 N장(N=GRID_COUNT)으로 고정
                            assert getattr(self, "val_dl", None) is not None, "val_dl이 없습니다. eval_use_val_split=True인지 확인하세요."
                            N = GRID_COUNT
                            xs, cs = [], []
                            val_it = iter(self.val_dl)   # val_dl: batch_size=1, shuffle=False
                            for _ in range(N):
                                batch = next(val_it)
                                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                                    x, c = batch[0], batch[1]
                                elif isinstance(batch, dict) and 'tof' in batch and 'mgre' in batch:
                                    x, c = batch['tof'], batch['mgre']
                                else:
                                    x, c = batch, None
                                xs.append(x)
                                if c is not None:
                                    cs.append(c)

                            vis_batch = torch.cat([t.to(self.device) for t in xs], dim=0)  # [N,C,H,W]

                            ### 2025-10-29 조건부 모델용 조건 텐서 병합 (이게 pretrian에서 동작할때 condition이 없어서 생기는 오류를 없애기 위한 코드임)
                            # vis_cond  = (torch.cat([t.to(self.device) for t in cs], dim=0) if (cs and len(cs)==len(xs)) else None)
                            vis_cond = None
                            if getattr(self.model, "use_control", False):
                                # 여기서만 vis_cond 만들기
                                if cs and len(cs) == len(xs) and isinstance(cs, (list, tuple)) and all(hasattr(t, "to") for t in cs):
                                    vis_cond = torch.cat([t.to(self.device) for t in cs], dim=0)


                            # print(vis_batch.shape, vis_batch.dtype, vis_batch.min().item(), vis_batch.max().item())
                            # print(vis_cond.shape,  vis_cond.dtype,  vis_cond.min().item(),  vis_cond.max().item()) if vis_cond is not None else None

                            # for iii in range(vis_cond.shape[1]) if vis_cond is not None else []:
                            #     print(vis_cond[:, iii, ...].shape, vis_cond[:, iii, ...].dtype, vis_cond[:, iii, ...].min().item(), vis_cond[:, iii, ...].max().item())

                            # if vis_cond is not None:
                            #     import matplotlib.pyplot as plt
                            #     conditions_dir = self.results_folder / 'conditions'
                            #     conditions_dir.mkdir(parents=True, exist_ok=True)
                            #     plt.figure(figsize=(12,6))
                            #     plt.imshow(vis_cond[0, 2].cpu(), cmap='gray')
                            #     plt.colorbar()
                            #     plt.savefig(self.results_folder / f'conditions/cond-sample-{milestone}.png', dpi=150)
                            #     plt.close()

                            B, C, H, W = vis_batch.shape
                            center_idx = (C - 1) // 2
                            # vis_batch = vis_batch[:min(B, GRID_COUNT)]  # 고정 N개를 이미 뽑았으므로 슬라이스 불필요
                            # if vis_cond is not None:
                            #     vis_cond = vis_cond[:vis_batch.size(0)]

                            #----------------------------2025.10.12 condition png 생성하기
                            # (1) 저장용으로 원본 cond 배치를 보존
                            vis_cond_for_save = vis_cond

                            # (2) mGRE가 [B,25,H,W] (= echo 5 * neighbor 5)라고 가정 → 첫 번째 에코의 5슬라이스만 [B,5,H,W]
                            def _first_echo_group(cond_tensor, neighbor_slices=5):
                                if cond_tensor is None:
                                    return None
                                Bc, Cc, Hc, Wc = cond_tensor.shape
                                if Cc < neighbor_slices:     # 안전장치
                                    return cond_tensor
                                return cond_tensor[:, 0:neighbor_slices, ...]   # 채널 0~4 = 1번째 에코의 5슬라이스
                            cond_5 = _first_echo_group(vis_cond_for_save, neighbor_slices=5)   # [B,5,H,W] or None
                            #----------------------------2025.10.12 condition png 생성하기 (끝)

                            # (C) 랜덤 샘플도 동일 그리드 크기로 (ControlNet cond 전달)
                            all_images_list = []
                            remain = GRID_COUNT

                            #----------------------------2025.10.12 condition png 생성하기
                            # 샘플링에만 사용할 포인터(이 변수만 재바인딩하여 '소모')
                            vis_cond_for_sampling = vis_cond
                            #----------------------------2025.10.12 condition png 생성하기 (끝)

                            while remain > 0:
                                n = min(self.batch_size, remain)
                                #----------------------------2025.10.12 condition png 생성하기
                                cond_slice = (vis_cond_for_sampling[:n] if vis_cond_for_sampling is not None else None)
                                #----------------------------2025.10.12 condition png 생성하기 (끝)
                                all_images_list.append(self.ema.ema_model.sample(batch_size=n, cond=cond_slice))  # [n,C,H,W], [0,1]
                                #----------------------------2025.10.12 condition png 생성하기
                                if vis_cond_for_sampling is not None:
                                    vis_cond_for_sampling = vis_cond_for_sampling[n:]
                                #----------------------------2025.10.12 condition png 생성하기 (끝)
                                remain -= n
                            all_images = torch.cat(all_images_list, dim=0)  # [GRID_COUNT, C, H, W]
                            
                            print("INPUT stats:", vis_batch.amin().item(), vis_batch.mean().item(), vis_batch.amax().item())
                            print("INPUT shape:", tuple(vis_batch.shape))
                            
                            print("SAMPLE stats:", all_images.amin().item(), all_images.mean().item(), all_images.amax().item())
                            print("SAMPLE shape:", tuple(all_images.shape))

                        # (D) 저장
                        inputs_dir  = self.results_folder / 'inputs'
                        samples_dir = self.results_folder / 'samples'
                        inputs_dir.mkdir(parents=True, exist_ok=True)
                        samples_dir.mkdir(parents=True, exist_ok=True)

                        #----------------------------2025.10.12 condition png 생성하기
                        # condition 전용 폴더
                        conditions_dir = self.results_folder / 'conditions'
                        conditions_dir.mkdir(parents=True, exist_ok=True)
                        print(conditions_dir)
                        #----------------------------2025.10.12 condition png 생성하기 (끝)


                        # (1) 중앙 채널만 저장 — 모두 n x n 그리드
                        if vis_save_mode in ("center", "center+all"):
                            vis_c = vis_batch[:, center_idx:center_idx+1]
                            sample_c = all_images[:, center_idx:center_idx+1]

                            if APPLY_STRETCH:
                                vis_c    = self._percentile_stretch(vis_c,    Q_LOW, Q_HIGH)
                                sample_c = self._percentile_stretch(sample_c, Q_LOW, Q_HIGH)

                            vis_c    = self._maybe_to_rgb(vis_c)
                            sample_c = self._maybe_to_rgb(sample_c)

                            utils.save_image(vis_c,    str(inputs_dir  / f'input-center-{milestone}.png'),  nrow=GRID_COLS, normalize=True, value_range=(-1,1), scale_each=True)
                            utils.save_image(sample_c, str(samples_dir / f'sample-center-{milestone}.png'), nrow=GRID_COLS, normalize=True, value_range=(-1,1), scale_each=True)

                            #----------------------------2025.10.12 condition png 생성하기
                            # cond_5([B,5,H,W])가 있으면 중앙 채널(5→idx=2)을 input/sample과 동일 파이프라인으로 저장
                            if cond_5 is not None and cond_5.shape[1] > 0:
                                cond_center_idx = (cond_5.shape[1] - 1) // 2    # 5채널이면 2
                                cond_c = cond_5[:, cond_center_idx:cond_center_idx+1]   # [B,1,H,W]
                                if APPLY_STRETCH:
                                    cond_c = self._percentile_stretch(cond_c, Q_LOW, Q_HIGH)
                                cond_c = self._maybe_to_rgb(cond_c)
                                utils.save_image(
                                    cond_c,
                                    str(conditions_dir / f'condition-center-{milestone}.png'),
                                    nrow=GRID_COLS, normalize=True, value_range=(-1,1), scale_each=True
                                )
                            #----------------------------2025.10.12 condition png 생성하기 (끝)



                        # # ===================== BEGIN PATCH (Seo 2025-09-30) =====================
                        # # ---------- 저장/시각화 설정 ----------
                        # n = int(math.sqrt(self.num_samples))
                        # GRID_COLS = GRID_ROWS = n
                        # GRID_COUNT = n * n

                        # # vis 저장 모드: "center" | "all" | "tiles" | "center+all"
                        # vis_save_mode = "center"

                        # # 입력 텐서의 범위
                        # VIS_INPUT_RANGE = 'minus1to1'

                        # APPLY_STRETCH = False
                        # Q_LOW, Q_HIGH = 0.01, 0.99
                        # replicate_to_rgb = False

                        # with torch.inference_mode():
                        #     milestone = self.step // self.save_and_sample_every

                        #     # (A) 시각화용 배치
                        #     vis_raw = next(self.dl)
                        #     if isinstance(vis_raw, (tuple, list)) and len(vis_raw) == 2:
                        #         vis_batch, vis_cond = vis_raw
                        #     elif isinstance(vis_raw, dict) and 'tof' in vis_raw and 'mgre' in vis_raw:
                        #         vis_batch, vis_cond = vis_raw['tof'], vis_raw['mgre']
                        #     else:
                        #         vis_batch, vis_cond = vis_raw, None

                        #     vis_batch = vis_batch.to(self.device)
                        #     vis_cond  = vis_cond.to(self.device) if vis_cond is not None else None

                        #     # print(vis_batch.shape, vis_batch.dtype, vis_batch.min().item(), vis_batch.max().item())
                        #     # print(vis_cond.shape,  vis_cond.dtype,  vis_cond.min().item(),  vis_cond.max().item()) if vis_cond is not None else None

                        #     # for iii in range(vis_cond.shape[1]) if vis_cond is not None else []:
                        #     #     print(vis_cond[:, iii, ...].shape, vis_cond[:, iii, ...].dtype, vis_cond[:, iii, ...].min().item(), vis_cond[:, iii, ...].max().item())

                        #     # if vis_cond is not None:
                        #     #     import matplotlib.pyplot as plt
                        #     #     conditions_dir = self.results_folder / 'conditions'
                        #     #     conditions_dir.mkdir(parents=True, exist_ok=True)
                        #     #     plt.figure(figsize=(12,6))
                        #     #     plt.imshow(vis_cond[0, 2].cpu(), cmap='gray')
                        #     #     plt.colorbar()
                        #     #     plt.savefig(self.results_folder / f'conditions/cond-sample-{milestone}.png', dpi=150)
                        #     #     plt.close()

                        #     B, C, H, W = vis_batch.shape
                        #     center_idx = (C - 1) // 2
                        #     vis_batch = vis_batch[:min(B, GRID_COUNT)]
                        #     if vis_cond is not None:
                        #         vis_cond = vis_cond[:vis_batch.size(0)]

                        #     #----------------------------2025.10.12 condition png 생성하기
                        #     # (1) 저장용으로 원본 cond 배치를 보존
                        #     vis_cond_for_save = vis_cond

                        #     # (2) mGRE가 [B,25,H,W] (= echo 5 * neighbor 5)라고 가정 → 첫 번째 에코의 5슬라이스만 [B,5,H,W]
                        #     def _first_echo_group(cond_tensor, neighbor_slices=5):
                        #         if cond_tensor is None:
                        #             return None
                        #         Bc, Cc, Hc, Wc = cond_tensor.shape
                        #         if Cc < neighbor_slices:     # 안전장치
                        #             return cond_tensor
                        #         return cond_tensor[:, 0:neighbor_slices, ...]   # 채널 0~4 = 1번째 에코의 5슬라이스
                        #     cond_5 = _first_echo_group(vis_cond_for_save, neighbor_slices=5)   # [B,5,H,W] or None
                        #     #----------------------------2025.10.12 condition png 생성하기 (끝)

                        #     # (C) 랜덤 샘플도 동일 그리드 크기로 (ControlNet cond 전달)
                        #     all_images_list = []
                        #     remain = GRID_COUNT

                        #     #----------------------------2025.10.12 condition png 생성하기
                        #     # 샘플링에만 사용할 포인터(이 변수만 재바인딩하여 '소모')
                        #     vis_cond_for_sampling = vis_cond
                        #     #----------------------------2025.10.12 condition png 생성하기 (끝)

                        #     while remain > 0:
                        #         n = min(self.batch_size, remain)
                        #         #----------------------------2025.10.12 condition png 생성하기
                        #         cond_slice = (vis_cond_for_sampling[:n] if vis_cond_for_sampling is not None else None)
                        #         #----------------------------2025.10.12 condition png 생성하기 (끝)
                        #         all_images_list.append(self.ema.ema_model.sample(batch_size=n, cond=cond_slice))  # [n,C,H,W], [0,1]
                        #         #----------------------------2025.10.12 condition png 생성하기
                        #         if vis_cond_for_sampling is not None:
                        #             vis_cond_for_sampling = vis_cond_for_sampling[n:]
                        #         #----------------------------2025.10.12 condition png 생성하기 (끝)
                        #         remain -= n
                        #     all_images = torch.cat(all_images_list, dim=0)  # [GRID_COUNT, C, H, W]
                            
                        #     print("INPUT stats:", vis_batch.amin().item(), vis_batch.mean().item(), vis_batch.amax().item())
                        #     print("INPUT shape:", tuple(vis_batch.shape))
                            
                        #     print("SAMPLE stats:", all_images.amin().item(), all_images.mean().item(), all_images.amax().item())
                        #     print("SAMPLE shape:", tuple(all_images.shape))

                        # # (D) 저장
                        # inputs_dir  = self.results_folder / 'inputs'
                        # samples_dir = self.results_folder / 'samples'
                        # inputs_dir.mkdir(parents=True, exist_ok=True)
                        # samples_dir.mkdir(parents=True, exist_ok=True)

                        # #----------------------------2025.10.12 condition png 생성하기
                        # # condition 전용 폴더
                        # conditions_dir = self.results_folder / 'conditions'
                        # conditions_dir.mkdir(parents=True, exist_ok=True)
                        # print(conditions_dir)
                        # #----------------------------2025.10.12 condition png 생성하기 (끝)


                        # # (1) 중앙 채널만 저장 — 모두 n x n 그리드
                        # if vis_save_mode in ("center", "center+all"):
                        #     vis_c = vis_batch[:, center_idx:center_idx+1]
                        #     sample_c = all_images[:, center_idx:center_idx+1]

                        #     if APPLY_STRETCH:
                        #         vis_c    = self._percentile_stretch(vis_c,    Q_LOW, Q_HIGH)
                        #         sample_c = self._percentile_stretch(sample_c, Q_LOW, Q_HIGH)

                        #     vis_c    = self._maybe_to_rgb(vis_c)
                        #     sample_c = self._maybe_to_rgb(sample_c)

                        #     utils.save_image(vis_c,    str(inputs_dir  / f'input-center-{milestone}.png'),  nrow=GRID_COLS, normalize=True, value_range=(-1,1), scale_each=True)
                        #     utils.save_image(sample_c, str(samples_dir / f'sample-center-{milestone}.png'), nrow=GRID_COLS, normalize=True, value_range=(-1,1), scale_each=True)

                        #     #----------------------------2025.10.12 condition png 생성하기
                        #     # cond_5([B,5,H,W])가 있으면 중앙 채널(5→idx=2)을 input/sample과 동일 파이프라인으로 저장
                        #     if cond_5 is not None and cond_5.shape[1] > 0:
                        #         cond_center_idx = (cond_5.shape[1] - 1) // 2    # 5채널이면 2
                        #         cond_c = cond_5[:, cond_center_idx:cond_center_idx+1]   # [B,1,H,W]
                        #         if APPLY_STRETCH:
                        #             cond_c = self._percentile_stretch(cond_c, Q_LOW, Q_HIGH)
                        #         cond_c = self._maybe_to_rgb(cond_c)
                        #         utils.save_image(
                        #             cond_c,
                        #             str(conditions_dir / f'condition-center-{milestone}.png'),
                        #             nrow=GRID_COLS, normalize=True, value_range=(-1,1), scale_each=True
                        #         )
                        #     #----------------------------2025.10.12 condition png 생성하기 (끝)
                            # ===================== END PATCH (Seo 2025-09-30) =====================








                            # ===================== END PATCH (Seo 2025-09-30) =====================

                            # whether to calculate fid
                            if self.calculate_fid:
                                fid_score = self.fid_scorer.fid_score()
                                accelerator.print(f'fid_score: {fid_score}')
                            else:
                                fid_score = None  # 👈 FID 비활성화 시 None

                            # ===================== 👇👇👇 새로 추가: val loss 평가 (구성요소별) 👇👇👇 =====================
                            val_stats = None  # {"val_total","val_base","val_mip","val_total_simple","val_combo"}
                            if self.use_val_loss_for_best:
                                try:
                                    # _quick_eval_loss 가 dict 또는 (avg, dict) 반환하는 경우 모두 지원
                                    _ret = self._quick_eval_loss()
                                    if isinstance(_ret, tuple):
                                        _, val_stats = _ret
                                    else:
                                        val_stats = _ret

                                    # total_simple이 없으면 base(+mip)로 계산
                                    if val_stats is not None and val_stats.get("val_total_simple") is None:
                                        vb = val_stats.get("val_base", None)
                                        vm = val_stats.get("val_mip",  0.0)
                                        if vb is not None:
                                            val_stats["val_total_simple"] = float(vb) + (0.0 if vm is None else float(vm))

                                    vt = val_stats.get("val_total", None) if val_stats else None
                                    vb = val_stats.get("val_base",  None) if val_stats else None
                                    vm = val_stats.get("val_mip",   None) if val_stats else None
                                    vs = val_stats.get("val_total_simple", None) if val_stats else None
                                    msg = [f"total={vt:.6f}" if vt is not None else "total=None"]
                                    if vb is not None: msg.append(f"base={vb:.6f}")
                                    if vm is not None: msg.append(f"mip={vm:.6f}")
                                    if vs is not None: msg.append(f"total_simple={vs:.6f}")
                                    accelerator.print(f'val (subset~{self.val_subset_batches} batches): ' + " ".join(msg))
                                except Exception as e:
                                    accelerator.print(f'val eval skipped due to error: {e}')
                                    val_stats = None
                            # ===================== 👆👆👆 새로 추가: val loss 평가 (구성요소별) 👆👆👆 =====================

                            # ===================== 👇👇👇 (ADD) TB: 검증 스칼라 로깅 👇👇👇 =====================
                            if getattr(self, "tb", None) is not None:
                                if fid_score is not None:
                                    self.tb.add_scalar("val/FID", float(fid_score), global_step=self.step)
                                if val_stats is not None:
                                    vt = val_stats.get("val_total", None)
                                    vb = val_stats.get("val_base",  None)
                                    vm = val_stats.get("val_mip",   None)
                                    vs = val_stats.get("val_total_simple", None)
                                    if vt is not None:
                                        # 호환성: 기존 "val/loss"는 total을 그대로 기록
                                        self.tb.add_scalar("val/loss",  float(vt), global_step=self.step)
                                        self.tb.add_scalar("val/total", float(vt), global_step=self.step)
                                    if vb is not None:
                                        self.tb.add_scalar("val/base",  float(vb), global_step=self.step)
                                    if vm is not None:
                                        self.tb.add_scalar("val/mip",   float(vm), global_step=self.step)
                                    if vs is not None:
                                        self.tb.add_scalar("val/total_simple", float(vs), global_step=self.step)
                            # ===================== 👆👆👆 (ADD) TB: 검증 스칼라 로깅 👆👆👆 =====================

                            if self.save_best_and_latest_only:
                                # ▼ 안전장치: policy/score 테이블이 없다면 여기서 1회 초기화
                                if not hasattr(self, "best_policy"):
                                    self.best_policy = "val_loss"
                                if not hasattr(self, "best_scores"):
                                    self.best_scores = {
                                        "FID": float("inf"),
                                        "val_loss": float("inf"),
                                        "base": float("inf"),
                                        "mip": float("inf"),
                                        "total": float("inf"),
                                        "total_simple": float("inf"),
                                        "combo": float("inf"),
                                    }

                                did_save_best = False
                                _improved = {}   # (ADD) 어떤 지표가 개선되었는지 기록

                                # latest는 항상 먼저 저장 + 메타 기록 준비
                                # JSON(info)에 넣을 extra payload (train/val 구성요소) 구성
                                def _to_float(x):
                                    if x is None: return None
                                    return float(x.item()) if hasattr(x, "item") else float(x)

                                train_extra = {}
                                if 'last_stats' in locals() and last_stats is not None:
                                    train_extra = {
                                        "train_base":  _to_float(last_stats.get("base")),
                                        "train_mip":   _to_float(last_stats.get("mip")),
                                        "train_total": _to_float(last_stats.get("total")),
                                    }

                                val_extra = {}
                                if val_stats is not None:
                                    val_extra = {
                                        "val_base":         None if val_stats.get("val_base")         is None else float(val_stats["val_base"]),
                                        "val_mip":          None if val_stats.get("val_mip")          is None else float(val_stats["val_mip"]),
                                        "val_total":        None if val_stats.get("val_total")        is None else float(val_stats["val_total"]),
                                        "val_total_simple": None if val_stats.get("val_total_simple") is None else float(val_stats["val_total_simple"]),
                                        "val_combo":        None if val_stats.get("val_combo")        is None else float(val_stats["val_combo"]),
                                    }

                                extra_payload = {}
                                extra_payload.update(train_extra)
                                extra_payload.update(val_extra)

                                self.save("latest")
                                try:
                                    self._record_ckpt_info(
                                        "latest",
                                        step=self.step,
                                        epoch=getattr(self, "epoch", None),
                                        metric_name=None,
                                        metric_value=None,
                                        extra=extra_payload,   # ★ 추가
                                    )
                                except Exception as e:
                                    accelerator.print(f"[meta] latest info write skipped: {e}")

                                # ---------- 후보 지표 구성 ----------
                                candidates = {}
                                if fid_score is not None:
                                    candidates["FID"] = float(fid_score)
                                # val_loss(=total)을 기본 후보로
                                vt = val_stats.get("val_total", None) if val_stats is not None else None
                                if vt is not None:
                                    candidates["val_loss"] = float(vt)
                                    candidates["total"]    = float(vt)
                                # 구성요소 후보들
                                if val_stats is not None:
                                    if val_stats.get("val_base") is not None:
                                        candidates["base"] = float(val_stats["val_base"])
                                    if val_stats.get("val_mip") is not None:
                                        candidates["mip"] = float(val_stats["val_mip"])
                                    if val_stats.get("val_total_simple") is not None:
                                        candidates["total_simple"] = float(val_stats["val_total_simple"])
                                    if val_stats.get("val_combo") is not None:
                                        candidates["combo"] = float(val_stats["val_combo"])

                                # ---------- 정책 결정 ----------
                                pol = (self.best_policy or "val_loss").lower()
                                if pol == "mip":
                                    keys = ["mip"]
                                elif pol == "base":
                                    keys = ["base"]
                                elif pol == "total_simple":
                                    keys = ["total_simple"]
                                elif pol == "val_loss":
                                    keys = ["val_loss"]
                                elif pol == "any":
                                    keys = [k for k in ("base", "mip", "total_simple") if k in candidates]
                                    for extra_k in ("total", "combo"):
                                        if extra_k in candidates:
                                            keys.append(extra_k)
                                else:
                                    keys = ["val_loss"]

                                # ---------- FID 개선 처리 (별도 best_FID.*) ----------
                                if "FID" in candidates and candidates["FID"] < self.best_scores["FID"]:
                                    self.best_scores["FID"] = candidates["FID"]
                                    self.save("best_FID")
                                    did_save_best = True
                                    _improved["FID"] = float(candidates["FID"])
                                    try:
                                        self._record_ckpt_info(
                                            "best_FID",
                                            step=self.step,
                                            epoch=getattr(self, "epoch", None),
                                            metric_name="FID",
                                            metric_value=float(candidates["FID"]),
                                            extra=extra_payload,
                                        )
                                    except Exception as e:
                                        accelerator.print(f"[meta] best_FID info write skipped: {e}")
                                    if getattr(self, "tb", None) is not None:
                                        self.tb.add_scalar("best/FID", float(candidates["FID"]), global_step=self.step)

                                # ---------- 정책 키들에 대해 각각 best_{k} 저장 ----------
                                for k in keys:
                                    if k not in candidates:
                                        continue
                                    cur = float(candidates[k])
                                    if cur < self.best_scores.get(k, float("inf")):
                                        self.best_scores[k] = cur
                                        milestone = f"best_{k}"  # model-best_{k}.*
                                        self.save(milestone)
                                        did_save_best = True
                                        _improved[k] = cur
                                        try:
                                            self._record_ckpt_info(
                                                milestone,
                                                step=self.step,
                                                epoch=getattr(self, "epoch", None),
                                                metric_name=k,
                                                metric_value=cur,
                                                extra=extra_payload,
                                            )
                                        except Exception as e:
                                            accelerator.print(f"[meta] {milestone} info write skipped: {e}")
                                        if getattr(self, "tb", None) is not None:
                                            self.tb.add_scalar(f"best/{k}", cur, global_step=self.step)

                                # ---------- TB 텍스트 요약 ----------
                                if did_save_best and getattr(self, "tb", None) is not None:
                                    vt_s = f"{val_stats['val_total']:.6f}"        if val_stats and val_stats.get('val_total')        is not None else "N/A"
                                    vb_s = f"{val_stats['val_base']:.6f}"         if val_stats and val_stats.get('val_base')         is not None else "N/A"
                                    vm_s = f"{val_stats['val_mip']:.6f}"          if val_stats and val_stats.get('val_mip')          is not None else "N/A"
                                    vs_s = f"{val_stats['val_total_simple']:.6f}" if val_stats and val_stats.get('val_total_simple') is not None else "N/A"
                                    improved_txt = ", ".join(f"{k}={_improved[k]:.6f}" for k in sorted(_improved.keys()))
                                    self.tb.add_text(
                                        "best/info",
                                        f"step={self.step}, val_total={vt_s}, val_base={vb_s}, val_mip={vm_s}, val_total_simple={vs_s} | improved: {improved_txt}",
                                        global_step=self.step
                                    )
                                # (ADD) best/latest 처리 끝난 뒤에도 주기 번호 ckp 추가 저장
                                self.save(str(self.step // self.save_and_sample_every))
                            else:
                                self.save(milestone)




    def sample_on_val(
        self,
        out_dir: Optional[str] = None,
        max_items: Optional[int] = None,
        vis_save_mode: str = "center",   # "center" | "center+all" (필요시 확장)
        apply_stretch: bool = False,
        q_low: float = 0.01,
        q_high: float = 0.99,
    ):
        """
        검증 데이터스트림(self.val_dl)을 순회하며 입력/조건/생성 이미지를 저장.
        - EMA 가중치 사용 (self.ema.ema_model)
        - batch_size=1, shuffle=False 전제
        """
        import math, torch
        from torchvision import utils

        assert self.val_dl is not None, "val_dl 이 없습니다. Trainer 생성 시 eval_use_val_split=True인지 확인하세요."
        ema_model = self.ema.ema_model if hasattr(self, "ema") else self.model
        ema_model.eval()

        device = self.device
        results_dir = self.results_folder
        save_root = (results_dir / "val_samples") if out_dir is None else Path(out_dir)
        inputs_dir  = save_root / "inputs"
        samples_dir = save_root / "samples"
        cond_dir    = save_root / "conditions"
        for d in (inputs_dir, samples_dir, cond_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 시각화 헬퍼 (학습 루프와 동일 정책)
        def _maybe_to_rgb(x):
            if x.dim() != 4: raise ValueError("expect [B,C,H,W]")
            return x.repeat(1,3,1,1) if x.size(1) == 1 else x

        def _percentile_stretch(x, lo, hi):
            # x: [B,C,H,W], per-image/channel 스트레치 (간단버전)
            B,C,H,W = x.shape
            y = []
            for b in range(B):
                yc = []
                for c in range(C):
                    v = x[b,c]
                    lo_v = torch.quantile(v.flatten(), lo)
                    hi_v = torch.quantile(v.flatten(), hi)
                    vv = (v - lo_v) / max(hi_v - lo_v, 1e-6)
                    yc.append(vv.clamp(0,1)[None])
                y.append(torch.cat(yc, dim=0)[None])
            return torch.cat(y, dim=0)

        # 메타를 돌려주는지 감지 (paired val 은 기본 False라 index 기반 이름을 씁니다)
        return_meta = False
        try:
            first = next(iter(self.val_dl))
            # 형태 복원
            if isinstance(first, (list, tuple)):
                maybe = first[0]
            else:
                maybe = first
            return_meta = isinstance(maybe, (tuple, list)) and len(maybe) == 2 and isinstance(maybe[1], dict)
        except Exception:
            pass

        # 다시 iterator 원복
        val_iter = iter(self.val_dl)

        with torch.inference_mode():
            for idx in range(10**9):
                if max_items is not None and idx >= max_items:
                    break

                batch = next(val_iter, None)
                if batch is None:
                    break

                # -------- 배치 파싱 (tof, mgre, meta) --------
                if return_meta:
                    (tof, mgre), meta = batch
                else:
                    meta = None
                    tof, mgre = batch, None
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        tof, mgre = batch
                    elif isinstance(batch, dict) and 'tof' in batch and 'mgre' in batch:
                        tof, mgre = batch['tof'], batch['mgre']

                tof  = tof.to(device)
                # mgre = mgre.to(device) if mgre is not None else None
                mgre = (mgre.to(device) if (mgre is not None and torch.is_tensor(mgre)) else None)
                print(tof.shape, tof.dtype, tof.min().item(), tof.max().item())
                print(mgre.shape, mgre.dtype, mgre.min().item(), mgre.max().item()) if mgre is not None else None

                B, C, H, W = tof.shape
                assert B == 1, "val_dl은 batch_size=1 가정입니다."  # 생성부가 간단해짐
                center_idx = (C - 1) // 2

                # -------- 생성 --------
                # cond는 그대로 전달(학습 루프와 동일)
                sample = ema_model.sample(batch_size=B, cond=mgre)  # [B,C,H,W], [0,1] 가정

                # -------- 저장 이름 구성 --------
                if meta is not None:
                    subj = meta.get("subject", "subj")
                    z    = meta.get("z", idx)
                    tag  = f"{subj}-z{int(z):04d}"
                else:
                    tag = f"idx{idx:05d}"

                # -------- 센터 채널 저장 (입력/샘플/조건) --------
                vis_c = tof[:, center_idx:center_idx+1]      # 입력은 [-1,1]일 수 있음
                samp_c = sample[:, center_idx:center_idx+1]  # 샘플은 [0,1] 가정

                # ========== ✅ NPY 저장(원본 텐서, 스트레치/컬러 변환 이전) ==========
                if self.accelerator.is_main_process:
                    # 풀 채널(원본)
                    np.save(inputs_dir  / f"val-input-full-{tag}.npy",  tof.detach().cpu().numpy().astype('float32'))
                    np.save(samples_dir / f"val-sample-full-{tag}.npy", sample.detach().cpu().numpy().astype('float32'))
                    # 센터 채널(원본)
                    np.save(inputs_dir  / f"val-input-center-{tag}.npy",  vis_c.detach().cpu().numpy().astype('float32'))
                    np.save(samples_dir / f"val-sample-center-{tag}.npy", samp_c.detach().cpu().numpy().astype('float32'))
                # =====================================================================

                if apply_stretch:
                    vis_c  = _percentile_stretch(vis_c,  q_low, q_high)
                    samp_c = _percentile_stretch(samp_c, q_low, q_high)

                vis_c  = _maybe_to_rgb(vis_c)
                samp_c = _maybe_to_rgb(samp_c)

                # 입력은 [-1,1] 스케일 가능성 → 학습 루프와 동일하게 normalize/value_range로 저장
                utils.save_image(
                    vis_c, str(inputs_dir / f"val-input-center-{tag}.png"),
                    nrow=1, normalize=True, value_range=(-1,1), scale_each=True
                )
                # 샘플은 [0,1] 가정이지만 동일 파이프라인 유지(학습과 같은 룰)
                utils.save_image(
                    samp_c, str(samples_dir / f"val-sample-center-{tag}.png"),
                    nrow=1, normalize=True, value_range=(-1,1), scale_each=True
                )

                # cond가 5-채널 블록(첫 에코의 5 이웃)이라면 중앙만 저장
                if mgre is not None and mgre.size(1) >= 5:
                    cond_5 = mgre[:, :5, ...]
                    cond_full = mgre[:, :, ...]
                    print('cond_5:', cond_5.shape, cond_5.dtype, cond_5.min().item(), cond_5.max().item())
                    print('cond_full:', cond_full.shape, cond_full.dtype, cond_full.min().item(), cond_full.max().item())
                    
                    cond_center_idx = (cond_5.shape[1] - 1) // 2
                    cond_c = cond_5[:, cond_center_idx:cond_center_idx+1]

                    # ===== ✅ NPY 저장(조건, 전처리 이전) =====
                    if self.accelerator.is_main_process:
                        np.save(cond_dir / f"val-condition-slice-full-{tag}.npy",
                                cond_full.detach().cpu().numpy().astype('float32'))
                        np.save(cond_dir / f"val-condition-center-{tag}.npy",
                                cond_c.detach().cpu().numpy().astype('float32'))
                    # =========================================

                    if apply_stretch: # 
                        cond_c = _percentile_stretch(cond_c, q_low, q_high)
                    cond_c = _maybe_to_rgb(cond_c)
                    utils.save_image(
                        cond_c, str(cond_dir / f"val-condition-center-{tag}.png"),
                        nrow=1, normalize=True, value_range=(-1,1), scale_each=True
                    )

        # 멀티프로세스 중복 저장 방지: 저장 자체는 main 프로세스 권장
        # (위 코드를 전체를 if self.accelerator.is_main_process: 블록으로 감싸도 됩니다)



    #----------수정 2025-10-20: 테스트 셋 샘플링 함수 추가(검증과 동일 정책)
    def sample_on_test(
        self,
        out_dir: Optional[str] = None,
        max_items: Optional[int] = None,
        vis_save_mode: str = "center",   # "center" | "center+all" (필요시 확장)
        apply_stretch: bool = False,
        q_low: float = 0.01,
        q_high: float = 0.99,
    ):
        """
        테스트 데이터스트림(self.test_dl)을 순회하며 입력/조건/생성 이미지를 저장.
        - EMA 가중치 사용 (self.ema.ema_model)
        - batch_size=1, shuffle=False 전제
        - sample_on_val과 동일 정책/구조, 저장 루트와 파일 접두어만 'test'로 변경
        """
        import math, torch
        from torchvision import utils

        assert getattr(self, "test_dl", None) is not None, "test_dl 이 없습니다. Trainer 생성 시 eval_use_test_split=True인지 확인하세요."
        ema_model = self.ema.ema_model if hasattr(self, "ema") else self.model
        ema_model.eval()

        device = self.device
        results_dir = self.results_folder
        save_root = (results_dir / "test_samples") if out_dir is None else Path(out_dir)   # ← test 전용 폴더
        inputs_dir  = save_root / "inputs"
        samples_dir = save_root / "samples"
        cond_dir    = save_root / "conditions"
        for d in (inputs_dir, samples_dir, cond_dir):
            d.mkdir(parents=True, exist_ok=True)

        # 시각화 헬퍼 (val과 동일)
        def _maybe_to_rgb(x):
            if x.dim() != 4: raise ValueError("expect [B,C,H,W]")
            return x.repeat(1,3,1,1) if x.size(1) == 1 else x

        def _percentile_stretch(x, lo, hi):
            # x: [B,C,H,W], per-image/channel 스트레치 (간단버전)
            B,C,H,W = x.shape
            y = []
            for b in range(B):
                yc = []
                for c in range(C):
                    v = x[b,c]
                    lo_v = torch.quantile(v.flatten(), lo)
                    hi_v = torch.quantile(v.flatten(), hi)
                    vv = (v - lo_v) / max(hi_v - lo_v, 1e-6)
                    yc.append(vv.clamp(0,1)[None])
                y.append(torch.cat(yc, dim=0)[None])
            return torch.cat(y, dim=0)

        # 메타 반환 감지
        return_meta = False
        try:
            first = next(iter(self.test_dl))
            maybe = first[0] if isinstance(first, (list, tuple)) else first
            return_meta = isinstance(maybe, (tuple, list)) and len(maybe) == 2 and isinstance(maybe[1], dict)
        except Exception:
            pass

        test_iter = iter(self.test_dl)

        with torch.inference_mode():
            for idx in range(10**9):
                if max_items is not None and idx >= max_items:
                    break

                batch = next(test_iter, None)
                if batch is None:
                    break

                # -------- 배치 파싱 (tof, mgre, meta) --------
                if return_meta:
                    (tof, mgre), meta = batch
                else:
                    meta = None
                    tof, mgre = batch, None
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        tof, mgre = batch
                    elif isinstance(batch, dict) and 'tof' in batch and 'mgre' in batch:
                        tof, mgre = batch['tof'], batch['mgre']

                tof  = tof.to(device)
                # mgre = mgre.to(device) if mgre is not None else None
                mgre = (mgre.to(device) if (mgre is not None and torch.is_tensor(mgre)) else None)
                print(tof.shape, tof.dtype, tof.min().item(), tof.max().item())
                print(mgre.shape, mgre.dtype, mgre.min().item(), mgre.max().item()) if mgre is not None else None

                B, C, H, W = tof.shape
                assert B == 1, "test_dl은 batch_size=1 가정입니다."
                center_idx = (C - 1) // 2

                # -------- 생성 --------
                #-------------------모든샘플저장하기 (시작)
                sample_all = ema_model.sample(batch_size=B, cond=mgre, return_all_timesteps=True)  # [B,T,C,H,W]
                sample = sample_all[:, -1, ...]  # [B,C,H,W]
                #-------------------모든샘플저장하기 (끝)

                # -------- 저장 이름 구성 --------
                if meta is not None:
                    subj = meta.get("subject", "subj")
                    z    = meta.get("z", idx)
                    tag  = f"{subj}-z{int(z):04d}"
                else:
                    tag = f"idx{idx:05d}"

                #-------------------포워드노이즈시각화 (시작)
                # 입력/조건의 정방향(포워드) 노이즈 과정을 같은 스케줄로 시각화 (t001..tT)
                try:
                    # alphas_cumprod 준비
                    if hasattr(ema_model, "alphas_cumprod") and ema_model.alphas_cumprod is not None:
                        acp = ema_model.alphas_cumprod.to(device)                 # [num_train_steps]
                    elif hasattr(ema_model, "betas") and ema_model.betas is not None:
                        betas = ema_model.betas.to(device)
                        acp = torch.cumprod(1. - betas, dim=0)                    # [num_train_steps]
                    else:
                        acp = None

                    T = sample_all.shape[1] - 1                                    # 초기노이즈 제외한 길이(예: 250)

                    # DDIM 서브스텝 매핑 가져오기 (가능하면)
                    if hasattr(ema_model, "ddim_timesteps") and ema_model.ddim_timesteps is not None:
                        t_indices = ema_model.ddim_timesteps.to(device).long()
                        if t_indices.numel() != T:
                            num_train = int(getattr(ema_model, "num_timesteps", acp.numel() if acp is not None else T))
                            t_indices = torch.linspace(1, num_train - 1, steps=T, device=device).long()
                    else:
                        num_train = int(getattr(ema_model, "num_timesteps", acp.numel() if acp is not None else T))
                        t_indices = torch.linspace(1, num_train - 1, steps=T, device=device).long()

                    if acp is not None and self.accelerator.is_main_process:
                        #-------------------포워드노이즈시각화 (수정: torch 버전 호환 난수)
                        def _seed_global(seed: int, is_cuda: bool):
                            if is_cuda:
                                torch.cuda.manual_seed_all(seed)
                            else:
                                torch.manual_seed(seed)

                        # 1) 입력(센터 채널) 포워드
                        fwd_in_dir = (inputs_dir / f"{tag}_forward")
                        fwd_in_dir.mkdir(parents=True, exist_ok=True)

                        x0 = tof[:, center_idx:center_idx+1].detach()             # [B,1,H,W]  ([-1,1] 가정)
                        seed_x = int(str(abs(hash(tag)))[:8], 10)
                        _seed_global(seed_x, x0.is_cuda)
                        eps = torch.randn_like(x0)                                 # <-- generator 인자 없이 전역 시드 사용

                        for k, t in enumerate(t_indices.tolist(), start=1):
                            at = acp[t].sqrt().view(1,1,1,1)
                            nt = (1. - acp[t]).sqrt().view(1,1,1,1)
                            xt = at * x0 + nt * eps                                # [B,1,H,W]
                            xvis = _percentile_stretch(xt, q_low, q_high) if apply_stretch else xt
                            xvis = _maybe_to_rgb(xvis)
                            utils.save_image(
                                xvis, str(fwd_in_dir / f"t{k:03d}.png"),
                                nrow=1, normalize=True, value_range=(-1,1), scale_each=True
                            )

                        # 2) 컨디션(센터 채널) 포워드 (있을 때)
                        if (mgre is not None) and (mgre.size(1) >= 1):
                            fwd_cond_dir = (cond_dir / f"{tag}_forward")
                            fwd_cond_dir.mkdir(parents=True, exist_ok=True)

                            if mgre.size(1) >= 5:
                                cond_center_idx = (min(5, mgre.size(1)) - 1) // 2
                            else:
                                cond_center_idx = (mgre.size(1) - 1) // 2
                            c0 = mgre[:, cond_center_idx:cond_center_idx+1].detach()  # [B,1,H,W]

                            seed_c = int(str(abs(hash(tag + "_cond")))[:8], 10)
                            _seed_global(seed_c, c0.is_cuda)
                            eps2 = torch.randn_like(c0)                               # <-- 동일하게 전역 시드 사용

                            for k, t in enumerate(t_indices.tolist(), start=1):
                                at = acp[t].sqrt().view(1,1,1,1)
                                nt = (1. - acp[t]).sqrt().view(1,1,1,1)
                                ct = at * c0 + nt * eps2                              # [B,1,H,W]
                                cvis = _percentile_stretch(ct, q_low, q_high) if apply_stretch else ct
                                cvis = _maybe_to_rgb(cvis)
                                utils.save_image(
                                    cvis, str(fwd_cond_dir / f"t{k:03d}.png"),
                                    nrow=1, normalize=True, value_range=(-1,1), scale_each=True
                                )
                except Exception as e:
                    # 포워드 시각화는 보조 기능이므로 실패해도 진행
                    print(f"[Forward-Vis skipped] {e}")
                #-------------------포워드노이즈시각화 (끝)

                #-------------------모든샘플저장하기 (계속)
                if self.accelerator.is_main_process:
                    step_out_dir = (samples_dir / f"{tag}_steps")
                    step_out_dir.mkdir(parents=True, exist_ok=True)
                    for t in range(1, sample_all.shape[1]):  # 초기노이즈(0) 제외 → 정확히 250장
                        # !! 배치 차원 유지 !!
                        step_c = sample_all[:, t, center_idx:center_idx+1, ...]  # [B,1,H,W]
                        if apply_stretch:
                            step_c = _percentile_stretch(step_c, q_low, q_high)
                        step_c = _maybe_to_rgb(step_c)  # [B,3,H,W]
                        utils.save_image(
                            step_c, str(step_out_dir / f"t{t:03d}.png"),
                            nrow=1, normalize=True, value_range=(-1,1)  # 최종 저장과 동일 스케일
                        )
                #-------------------모든샘플저장하기 (끝)

                # -------- 센터 채널 저장 (입력/샘플/조건) --------
                vis_c  = tof[:, center_idx:center_idx+1]
                samp_c = sample[:, center_idx:center_idx+1]

                # ========== ✅ NPY 저장(원본 텐서, 스트레치/컬러 변환 이전) ==========
                if self.accelerator.is_main_process:
                    np.save(inputs_dir  / f"test-input-full-{tag}.npy",   tof.detach().cpu().numpy().astype('float32'))      # ← 'test-' 접두어
                    np.save(samples_dir / f"test-sample-full-{tag}.npy",  sample.detach().cpu().numpy().astype('float32'))   # ← 'test-' 접두어
                    np.save(inputs_dir  / f"test-input-center-{tag}.npy",  vis_c.detach().cpu().numpy().astype('float32'))   # ← 'test-' 접두어
                    np.save(samples_dir / f"test-sample-center-{tag}.npy", samp_c.detach().cpu().numpy().astype('float32'))  # ← 'test-' 접두어
                # ====================================================================

                if apply_stretch: 
                    vis_c  = _percentile_stretch(vis_c,  q_low, q_high)
                    samp_c = _percentile_stretch(samp_c, q_low, q_high)

                vis_c  = _maybe_to_rgb(vis_c)
                samp_c = _maybe_to_rgb(samp_c)

                utils.save_image(
                    vis_c, str(inputs_dir / f"test-input-center-{tag}.png"),
                    nrow=1, normalize=True, value_range=(-1,1), scale_each=True
                )
                utils.save_image(
                    samp_c, str(samples_dir / f"test-sample-center-{tag}.png"),
                    nrow=1, normalize=True, value_range=(-1,1), scale_each=True
                )

                # cond 저장(있다면)
                if mgre is not None and mgre.size(1) >= 5:
                    cond_5 = mgre[:, :5, ...]
                    cond_full = mgre[:, :, ...]
                    print('cond_5:', cond_5.shape, cond_5.dtype, cond_5.min().item(), cond_5.max().item())
                    print('cond_full:', cond_full.shape, cond_full.dtype, cond_full.min().item(), cond_full.max().item())
                    
                    cond_center_idx = (cond_5.shape[1] - 1) // 2
                    cond_c = cond_5[:, cond_center_idx:cond_center_idx+1]

                    # ===== ✅ NPY 저장(조건, 전처리 이전) =====
                    if self.accelerator.is_main_process:
                        np.save(cond_dir / f"test-condition-slice-full-{tag}.npy",
                                cond_full.detach().cpu().numpy().astype('float32'))            # ← 'test-' 접두어
                        np.save(cond_dir / f"test-condition-center-{tag}.npy",
                                cond_c.detach().cpu().numpy().astype('float32'))               # ← 'test-' 접두어
                    # =========================================

                    if apply_stretch:
                        cond_c = _percentile_stretch(cond_c, q_low, q_high)
                    cond_c = _maybe_to_rgb(cond_c)
                    utils.save_image(
                        cond_c, str(cond_dir / f"test-condition-center-{tag}.png"),
                        nrow=1, normalize=True, value_range=(-1,1), scale_each=True
                    )
    #----------수정 2025-10-20 끝




    # #----------수정 2025-10-20: 테스트 셋 샘플링 함수 추가(검증과 동일 정책)
    # def sample_on_test(
    #     self,
    #     out_dir: Optional[str] = None,
    #     max_items: Optional[int] = None,
    #     vis_save_mode: str = "center",   # "center" | "center+all" (필요시 확장)
    #     apply_stretch: bool = False,
    #     q_low: float = 0.01,
    #     q_high: float = 0.99,
    # ):
    #     """
    #     테스트 데이터스트림(self.test_dl)을 순회하며 입력/조건/생성 이미지를 저장.
    #     - EMA 가중치 사용 (self.ema.ema_model)
    #     - batch_size=1, shuffle=False 전제
    #     - sample_on_val과 동일 정책/구조, 저장 루트와 파일 접두어만 'test'로 변경
    #     """
    #     import math, torch
    #     from torchvision import utils

    #     assert getattr(self, "test_dl", None) is not None, "test_dl 이 없습니다. Trainer 생성 시 eval_use_test_split=True인지 확인하세요."
    #     ema_model = self.ema.ema_model if hasattr(self, "ema") else self.model
    #     ema_model.eval()

    #     device = self.device
    #     results_dir = self.results_folder
    #     save_root = (results_dir / "test_samples") if out_dir is None else Path(out_dir)   # ← test 전용 폴더
    #     inputs_dir  = save_root / "inputs"
    #     samples_dir = save_root / "samples"
    #     cond_dir    = save_root / "conditions"
    #     for d in (inputs_dir, samples_dir, cond_dir):
    #         d.mkdir(parents=True, exist_ok=True)

    #     # 시각화 헬퍼 (val과 동일)
    #     def _maybe_to_rgb(x):
    #         if x.dim() != 4: raise ValueError("expect [B,C,H,W]")
    #         return x.repeat(1,3,1,1) if x.size(1) == 1 else x

    #     def _percentile_stretch(x, lo, hi):
    #         # x: [B,C,H,W], per-image/channel 스트레치 (간단버전)
    #         B,C,H,W = x.shape
    #         y = []
    #         for b in range(B):
    #             yc = []
    #             for c in range(C):
    #                 v = x[b,c]
    #                 lo_v = torch.quantile(v.flatten(), lo)
    #                 hi_v = torch.quantile(v.flatten(), hi)
    #                 vv = (v - lo_v) / max(hi_v - lo_v, 1e-6)
    #                 yc.append(vv.clamp(0,1)[None])
    #             y.append(torch.cat(yc, dim=0)[None])
    #         return torch.cat(y, dim=0)

    #     # 메타 반환 감지
    #     return_meta = False
    #     try:
    #         first = next(iter(self.test_dl))
    #         maybe = first[0] if isinstance(first, (list, tuple)) else first
    #         return_meta = isinstance(maybe, (tuple, list)) and len(maybe) == 2 and isinstance(maybe[1], dict)
    #     except Exception:
    #         pass

    #     test_iter = iter(self.test_dl)

    #     with torch.inference_mode():
    #         for idx in range(10**9):
    #             if max_items is not None and idx >= max_items:
    #                 break

    #             batch = next(test_iter, None)
    #             if batch is None:
    #                 break

    #             # -------- 배치 파싱 (tof, mgre, meta) --------
    #             if return_meta:
    #                 (tof, mgre), meta = batch
    #             else:
    #                 meta = None
    #                 tof, mgre = batch, None
    #                 if isinstance(batch, (tuple, list)) and len(batch) == 2:
    #                     tof, mgre = batch
    #                 elif isinstance(batch, dict) and 'tof' in batch and 'mgre' in batch:
    #                     tof, mgre = batch['tof'], batch['mgre']

    #             tof  = tof.to(device)
    #             # mgre = mgre.to(device) if mgre is not None else None
    #             mgre = (mgre.to(device) if (mgre is not None and torch.is_tensor(mgre)) else None)
    #             print(tof.shape, tof.dtype, tof.min().item(), tof.max().item())
    #             print(mgre.shape, mgre.dtype, mgre.min().item(), mgre.max().item()) if mgre is not None else None

    #             B, C, H, W = tof.shape
    #             assert B == 1, "test_dl은 batch_size=1 가정입니다."
    #             center_idx = (C - 1) // 2

    #             # -------- 생성 --------
    #             sample = ema_model.sample(batch_size=B, cond=mgre)  # [B,C,H,W], [0,1] 가정

    #             # -------- 저장 이름 구성 --------
    #             if meta is not None:
    #                 subj = meta.get("subject", "subj")
    #                 z    = meta.get("z", idx)
    #                 tag  = f"{subj}-z{int(z):04d}"
    #             else:
    #                 tag = f"idx{idx:05d}"

    #             # -------- 센터 채널 저장 (입력/샘플/조건) --------
    #             vis_c  = tof[:, center_idx:center_idx+1]
    #             samp_c = sample[:, center_idx:center_idx+1]

    #             # ========== ✅ NPY 저장(원본 텐서, 스트레치/컬러 변환 이전) ==========
    #             if self.accelerator.is_main_process:
    #                 # 풀 채널(원본)
    #                 np.save(inputs_dir  / f"test-input-full-{tag}.npy",   tof.detach().cpu().numpy().astype('float32'))      # ← 'test-' 접두어
    #                 np.save(samples_dir / f"test-sample-full-{tag}.npy",  sample.detach().cpu().numpy().astype('float32'))   # ← 'test-' 접두어
    #                 # 센터 채널(원본)
    #                 np.save(inputs_dir  / f"test-input-center-{tag}.npy",  vis_c.detach().cpu().numpy().astype('float32'))   # ← 'test-' 접두어
    #                 np.save(samples_dir / f"test-sample-center-{tag}.npy", samp_c.detach().cpu().numpy().astype('float32'))  # ← 'test-' 접두어
    #             # ====================================================================

    #             if apply_stretch:
    #                 vis_c  = _percentile_stretch(vis_c,  q_low, q_high)
    #                 samp_c = _percentile_stretch(samp_c, q_low, q_high)

    #             vis_c  = _maybe_to_rgb(vis_c)
    #             samp_c = _maybe_to_rgb(samp_c)

    #             # 입력은 [-1,1] 스케일 가능성 → val과 동일 파이프라인 유지
    #             utils.save_image(
    #                 vis_c, str(inputs_dir / f"test-input-center-{tag}.png"),    # ← 'test-' 접두어
    #                 nrow=1, normalize=True, value_range=(-1,1), scale_each=True
    #             )
    #             utils.save_image(
    #                 samp_c, str(samples_dir / f"test-sample-center-{tag}.png"), # ← 'test-' 접두어
    #                 nrow=1, normalize=True, value_range=(-1,1), scale_each=True
    #             )

    #             # cond 저장(있다면)
    #             if mgre is not None and mgre.size(1) >= 5:
    #                 cond_5 = mgre[:, :5, ...]
    #                 cond_full = mgre[:, :, ...]
    #                 print('cond_5:', cond_5.shape, cond_5.dtype, cond_5.min().item(), cond_5.max().item())
    #                 print('cond_full:', cond_full.shape, cond_full.dtype, cond_full.min().item(), cond_full.max().item())
                    
    #                 cond_center_idx = (cond_5.shape[1] - 1) // 2
    #                 cond_c = cond_5[:, cond_center_idx:cond_center_idx+1]

    #                 # ===== ✅ NPY 저장(조건, 전처리 이전) =====
    #                 if self.accelerator.is_main_process:
    #                     np.save(cond_dir / f"test-condition-slice-full-{tag}.npy",
    #                             cond_full.detach().cpu().numpy().astype('float32'))            # ← 'test-' 접두어
    #                     np.save(cond_dir / f"test-condition-center-{tag}.npy",
    #                             cond_c.detach().cpu().numpy().astype('float32'))               # ← 'test-' 접두어
    #                 # =========================================

    #                 if apply_stretch:
    #                     cond_c = _percentile_stretch(cond_c, q_low, q_high)
    #                 cond_c = _maybe_to_rgb(cond_c)
    #                 utils.save_image(
    #                     cond_c, str(cond_dir / f"test-condition-center-{tag}.png"),           # ← 'test-' 접두어
    #                     nrow=1, normalize=True, value_range=(-1,1), scale_each=True
    #                 )
    # #----------수정 2025-10-20 끝
