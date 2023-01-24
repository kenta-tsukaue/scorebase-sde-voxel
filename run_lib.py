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
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
import numpy as np
import logging
from PIL import Image
# Keep the import below for registering all model definitions
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
import tensorflow as tf
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  # Initialize model.　モデルの生成
  score_model = mutils.create_model(config)
  """このemaとは何者なのか分からない"""
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate) #(ファイル先) ./models/ema.py 
  optimizer = losses.get_optimizer(config, score_model.parameters()) #(ファイル先) ./losses.py
  """このdictはdictionaryを作るpythonの関数"""
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory 
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators データセットを作成
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  
  
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs(SDEの設定)
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    print("yaa")
    #raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
  
  # Building sampling functions 
  if config.training.snapshot_sampling:
    #sampling_shape = (config.training.batch_size, config.data.num_channels,
    #                  config.data.image_size, config.data.image_size, config.data.image_size)
    sampling_shape = (100, config.data.num_channels,
                      config.data.image_size, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
  
  
    
  
  for step in range(0, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    #print(batch.shape)
    #batch = torch.where(batch > 0, 255, 128)
    """
    for i in range(32):
      print("===================[",i+1, "]チャネル目==================")
      for j in range(32):
        print(batch[0][i][j])"""
    
    if step == 1:
      #batch = batch.permute(0, 3, 1, 2)
      input_batch = np.clip(batch.cpu().numpy(), 0, 255).astype(np.uint8) #画像用に補正
      input_dir = "./input"
      for i in range(10):
        for j in range(32):
            Image.fromarray(input_batch[i][j]).save( input_dir + "/" + str(i+1) + "_channel" + str(j+1) + ".png")
    batch = scaler(batch)

    batch = torch.unsqueeze(batch,dim=-4)  #  add channel axis for debug
    #print(batch.shape)
    #バッチの詳細を表示するためのもの
    """
    if step == 1:
      for i in range(32):
        for j in range(32):
          print(batch[0][0][i][j])
    """
   # batch = scaler(batch)


    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      eval_batch = torch.unsqueeze(eval_batch,dim=-4)  #  add channel axis for debug
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if config.training.snapshot_sampling:
        print("サンプリングを行います")
        if config.data.dataset == "SLICE":
          ema.store(score_model.parameters())
          ema.copy_to(score_model.parameters())
          sample, n = sampling_fn(score_model)
          sample_np = sample.cpu().numpy()
          ema.restore(score_model.parameters())
          this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
          tf.io.gfile.makedirs(this_sample_dir)
          sample = np.clip(sample.cpu().numpy() * 255, 0, 255).astype(np.uint8) #画像用に補正
          #print(sample.shape) (32, 1, 32, 32, 32)

          for i in range(10):
            for j in range(32):
              Image.fromarray(sample[i][0][j]).save(this_sample_dir + "/" + str(i+1) + "_channel" + str(j+1) + ".png")

          for i in range(16):
            with tf.io.gfile.GFile(
                os.path.join(this_sample_dir, "sample" + str(i+1)+ ".np"), "wb") as fout:
              np.save(fout, sample_np[i][0])
        else:
          ema.store(score_model.parameters())
          ema.copy_to(score_model.parameters())
          sample, n = sampling_fn(score_model)
          ema.restore(score_model.parameters())
          this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
          tf.io.gfile.makedirs(this_sample_dir)
          nrow = int(np.sqrt(sample.shape[0]))
          image_grid = make_grid(sample, nrow, padding=2)
          sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
            np.save(fout, sample)

          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
            save_image(image_grid, fout)
        

    

