# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import sys
import torch
import torch.nn as nn
import functools

from . import utils, layers, normalization

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init



class DDPM(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf # 128
    ch_mult = config.model.ch_mult # (1, 2, 2, 2)
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks #2
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions #(16,)
#    print('attn_resolutions=',attn_resolutions)
    dropout = config.model.dropout # 0.1
    resamp_with_conv = config.model.resamp_with_conv # True
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional = config.model.conditional # True
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional: #True
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered #False
    channels = config.data.num_channels #1

    # Downsampling block
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):#8
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

  def forward(self, x, labels):
    modules = self.all_modules
    m_idx = 0
    if self.conditional: #True
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None
#    print('at 124 temb=',temb.shape)

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    # Downsampling block
#    print('at 132',h.shape)
    hs = [modules[m_idx](h)]
#    print('at 134',hs[-1].shape)
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
#        print('at 138',i_level,i_block,self.num_res_blocks,h.shape)
        m_idx += 1
#        if h.shape[-2] in self.attn_resolutions:  #  use y dim
        if h.shape[-3] in self.attn_resolutions:  #  use x dim
#          print('at 143',h.shape); sys.stdout.flush()
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
#        print('at 148',i_level,i_block,h.shape); sys.stdout.flush()
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
#    print('at 149',h.shape); sys.stdout.flush()
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        print('at 160',i_level,i_block,h.shape)
        print("hの形は" + str(h.shape))
        print("hs.popの形は" + str(hs.pop().shape))
        print("catは", torch.cat([h, hs.pop()], dim=1))
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
#      if h.shape[-2] in self.attn_resolutions:  #  use y dim
      if h.shape[-3] in self.attn_resolutions:  #  use x dim
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h
