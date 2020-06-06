import config

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


if config.USE_TPU:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.distributed.parallel_loader as pl
  import torch_xla.distributed.xla_multiprocessing as xmp
  


def train(train_loader, GAN_Model, netD, VGG_MODEL, optG, optD, device, losses):
  batch = 0

  def wgan_loss(prediction, real_or_not):
    if real_or_not:
      return -torch.mean(prediction.float())
    else:
      return torch.mean(prediction.float())

  def gp_loss(y_pred, averaged_samples, gradient_penalty_weight):

    gradients = torch.autograd.grad(y_pred,averaged_samples,
                              grad_outputs=torch.ones(y_pred.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = (((gradients+1e-16).norm(2, dim=1) - 1) ** 2).mean() * gradient_penalty_weight
    return gradient_penalty
  for trainL, trainAB, _ in tqdm(iter(train_loader)):
      batch += 1  

      trainL_3 = torch.tensor(np.tile(trainL.cpu(), [1,3,1,1]), device=device)

      trainL = torch.tensor(trainL, device=device).double()
      trainAB = torch.tensor(trainAB, device=device).double()
      # trainL_3 = trainL_3.to(device).double()
      
      predictVGG = F.softmax(VGG_MODEL(trainL_3))

      ############ GAN MODEL ( Training Generator) ###################
      optG.zero_grad()
      predAB, classVector, discpred = GAN_Model(trainL, trainL_3)
      D_G_z1 = discpred.mean().item()
      Loss_KLD = nn.KLDivLoss(size_average='False')(classVector.log().float(), predictVGG.detach().float()) * 0.003
      Loss_MSE = nn.MSELoss()(predAB.float(), trainAB.float())
      Loss_WL = wgan_loss(discpred.float(), True) * 0.1 
      Loss_G = Loss_KLD + Loss_MSE + Loss_WL
      Loss_G.backward()

      if config.USE_TPU:
        if config.MULTI_CORE:
          xm.optimizer_step(optG)
        else:
          xm.optimizer_step(optG, barrier=True)
      else:
        optG.step()

      losses['G_losses'].append(Loss_G.item())
      losses['EPOCH_G_losses'].append(Loss_G.item())



      ################################################################

      ############### Discriminator Training #########################

      for param in netD.parameters():
        param.requires_grad = True

      optD.zero_grad()
      predLAB = torch.cat([trainL, predAB], dim=1)
      discpred = netD(predLAB.detach())
      D_G_z2 = discpred.mean().item()
      realLAB = torch.cat([trainL, trainAB], dim=1)
      discreal = netD(realLAB)
      D_x = discreal.mean().item()

      weights = torch.randn((trainAB.size(0),1,1,1) device=device)          
      averaged_samples = (weights * trainAB ) + ((1 - weights) * predAB.detach())
      averaged_samples = torch.autograd.Variable(averaged_samples, requires_grad=True)
      avg_img = torch.cat([trainL, averaged_samples], dim=1)
      discavg = netD(avg_img)

      Loss_D_Fake = wgan_loss(discpred, False)
      Loss_D_Real = wgan_loss(discreal, True)
      Loss_D_avg = gp_loss(discavg, averaged_samples, config.GRADIENT_PENALTY_WEIGHT)

      Loss_D = Loss_D_Fake + Loss_D_Real + Loss_D_avg
      Loss_D.backward()
      if config.USE_TPU:
        if config.MULTI_CORE:
          xm.optimzer_step(optD)
        else:
          xm.optimizer_step(optD, barrier=True)
      else:
        optD.step()

      losses['D_losses'].append(Loss_D.item())
      losses['EPOCH_D_losses'].append(Loss_D.item())
      # Output training stats
      if batch % 100 == 0:
        print('Loss_D: %.8f | Loss_G: %.8f | D(x): %.8f | D(G(z)): %.8f / %.8f | MSE: %.8f | KLD: %.8f | WGAN_F(G): %.8f | WGAN_F(D): %.8f | WGAN_R(D): %.8f | WGAN_A(D): %.8f'
            % (Loss_D.item(), Loss_G.item(), D_x, D_G_z1, D_G_z2,Loss_MSE.item(),Loss_KLD.item(),Loss_WL.item(), Loss_D_Fake.item(), Loss_D_Real.item(), Loss_D_avg.item()))

      