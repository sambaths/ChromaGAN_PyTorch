import model
import config
import dataset
import utils
import engine

import torch
import torchvision
import time

if config.USE_TPU:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.distributed.parallel_loader as pl
  import torch_xla.distributed.xla_multiprocessing as xmp
  

def map_fn(index=None, flags=None):
  torch.set_default_tensor_type('torch.FloatTensor')

  torch.manual_seed(flags['seed'])
  
  train_data = dataset.DATA(config.TRAIN_DIR) 

  if config.MULTI_CORE:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_data,
      num_replicas=xm.xrt_world_size(),
      rank=xm.get_ordinal(),
      shuffle=True)
  else:
    train_sampler = torch.utils.data.RandomSampler(train_data)

  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=flags['batch_size'] if config.MULTI_CORE else config.BATCH_SIZE,
      sampler=train_sampler,
      num_workers=flags['num_workers'] if config.MULTI_CORE else 1,
      drop_last=True)

  if config.MULTI_CORE:
    DEVICE = xm.xla_device()
  else:
    DEVICE = config.DEVICE


  netG = model.colorization_model().double()
  netD = model.discriminator_model().double()

  VGG_modelF = torchvision.models.vgg16(pretrained=True).double()
  VGG_modelF.requires_grad_(False)

  netG = netG.to(DEVICE)
  netD = netD.to(DEVICE)
  netGAN = model.GAN(netG, netD)
  VGG_modelF = VGG_modelF.to(DEVICE)

  optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
  optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
  ## Trains
  train_start = time.time()
  losses = {
      'G_losses' : [],
      'D_losses' : [],
      'EPOCH_G_losses' : [],
      'EPOCH_D_losses' : [],
      'G_losses_eval' : []
  }

  for epoch in range(flags['num_epochs'] if config.MULTI_CORE else config.NUM_EPOCHS):
    print('\n')
    print('#'*8,f'EPOCH-{epoch}','#'*8)
    losses['EPOCH_G_losses'] = []
    losses['EPOCH_D_losses'] = []
    if config.MULTI_CORE:
      para_train_loader = pl.ParallelLoader(train_loader, [DEVICE]).per_device_loader(DEVICE)
      engine.train(para_train_loader, netGAN, netD, VGG_modelF, optG, optD, device=DEVICE, losses=losses)
    else:
      engine.train(train_loader, netGAN, netD, VGG_modelF, optG, optD, device=DEVICE, losses=losses)
    elapsed_train_time = time.time() - train_start
    print("Process", index, "finished training. Train time was:", elapsed_train_time) 

    #########################CHECKPOINTING#################################
    checkpoint = {
          'epoch' : epoch,
          'generator_state_dict' :netG.state_dict(),
          'generator_optimizer': optG.state_dict(),
          'discriminator_state_dict': netD.state_dict(),
          'discriminator_optimizer': optD.state_dict()
      }
    xm.save(checkpoint, f'{config.CHECKPOINT_DIR}{epoch}_checkpoint.pt')
    del checkpoint
    ########################################################################
    utils.plot_some(train_data, netG, DEVICE)
# Configures training (and evaluation) parameters
flags = {}
flags['batch_size'] = config.BATCH_SIZE
flags['num_workers'] = 8
flags['num_epochs'] = config.NUM_EPOCHS
flags['seed'] = 1234


def run():
    if config.MULTI_CORE:
        xmp.spawn(map_fn, args=(flags,), nprocs=8, start_method='fork')
    else:
        map_fn()