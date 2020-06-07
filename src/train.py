
import model
import config
import dataset
import utils
import engine

import time
import torch
import torchvision
import warnings
warnings.filterwarnings('ignore')

import gc

if config.USE_TPU:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.distributed.parallel_loader as pl
  import torch_xla.distributed.xla_multiprocessing as xmp
  

def map_fn(index=None, flags=None):
  torch.set_default_tensor_type('torch.FloatTensor')
  torch.manual_seed(1234)
  
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
      num_workers=flags['num_workers'] if config.MULTI_CORE else 4,
      drop_last=True,
      pin_memory=True)

  if config.MULTI_CORE:
    DEVICE = xm.xla_device()
  else:
    DEVICE = config.DEVICE


  netG = model.colorization_model()
  netD = model.discriminator_model()

  VGG_modelF = torchvision.models.vgg16(pretrained=True)
  VGG_modelF.requires_grad_(False)

  netG = netG.to(DEVICE)
  netD = netD.to(DEVICE)
  
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

  netG, optG, netD, optD, epoch_checkpoint = utils.load_checkpoint(config.CHECKPOINT_DIR, netG, optG, netD, optD, DEVICE)
  netGAN = model.GAN(netG, netD)
  for epoch in range(epoch_checkpoint,flags['num_epochs']+1 if config.MULTI_CORE else config.NUM_EPOCHS+1):
    print('\n')
    print('#'*8,f'EPOCH-{epoch}','#'*8)
    losses['EPOCH_G_losses'] = []
    losses['EPOCH_D_losses'] = []
    if config.MULTI_CORE:
      para_train_loader = pl.ParallelLoader(train_loader, [DEVICE]).per_device_loader(DEVICE)
      engine.train(para_train_loader, netGAN, netD, VGG_modelF, optG, optD, device=DEVICE, losses=losses)
      elapsed_train_time = time.time() - train_start
      print("Process", index, "finished training. Train time was:", elapsed_train_time) 
    else:
      engine.train(train_loader, netGAN, netD, VGG_modelF, optG, optD, device=DEVICE, losses=losses)
    #########################CHECKPOINTING#################################
    utils.create_checkpoint(epoch, netG, optG, netD, optD, max_checkpoint=config.KEEP_CKPT, save_path = config.CHECKPOINT_DIR)
    ########################################################################
    utils.plot_some(train_data, netG, DEVICE, epoch)
    gc.collect()
# Configures training (and evaluation) parameters

def run():
  if config.MULTI_CORE:
      flags = {}
      flags['batch_size'] = config.BATCH_SIZE
      flags['num_workers'] = 8
      flags['num_epochs'] = config.NUM_EPOCHS
      flags['seed'] = 1234
      xmp.spawn(map_fn, args=(flags,), nprocs=8, start_method='fork')
  else:
      map_fn()
    # print(flags)
if __name__=='__main__':
  run()