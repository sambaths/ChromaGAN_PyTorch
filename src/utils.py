
import os
import cv2
import torch
import config
import glob
import numpy as np
import matplotlib.pyplot as plt

if config.USE_TPU:
  import torch_xla.core.xla_model as xm

def preprocess(imgs):
  try:
    imgs = imgs.detach().numpy()
  except:
    pass
  imgs = imgs * 255
  imgs[imgs>255] = 255
  imgs[imgs<0] = 0 
  return imgs.astype(np.uint8) # torch.unit8

def reconstruct(batchX, predictedY, filelist):

    batchX = batchX.reshape(224,224,1) 
    predictedY = predictedY.reshape(224,224,2)
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2RGB)
    save_results_path = config.OUT_DIR
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(save_results_path, filelist +  "_reconstructed.jpg" )
    cv2.imwrite(save_path, result)
    return result
    
def reconstruct_no(batchX, predictedY):

    batchX = batchX.reshape(224,224,1) 
    predictedY = predictedY.reshape(224,224,2)
    
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2RGB)
    return result


def imag_gird(axrow, orig, batchL, preds, epoch):
  fig , ax = plt.subplots(1,3, figsize=(15,15))
  ax[0].imshow(orig)
  ax[0].set_title('Original Image')

  ax[1].imshow(np.tile(batchL,(1,1,3)))
  ax[1].set_title('L Image with Channels reapeated(Input)') 

  ax[2].imshow(preds)
  ax[2].set_title('Pred Image')
  plt.savefig(f'sample_preds_{epoch}')
  # plt.show()

def plot_some(test_data, colorization_model, device, epoch):
  with torch.no_grad():
    indexes = [0, 2, 9]
    for idx in indexes:
    # for batch in range(TOTAL_TEST_BATCH):
      #torch.randint(0, len(test_data), (1,)).item()
      # idx=
      batchL, realAB, filename = test_data[idx]
      filepath = config.TRAIN_DIR+filename
      batchL = batchL.reshape(1,1,224,224)
      realAB = realAB.reshape(1,2,224,224)
      batchL_3 = torch.tensor(np.tile(batchL, [1, 3, 1, 1]))
      batchL_3 = batchL_3.to(device)
      batchL = torch.tensor(batchL).to(device).double()
      realAB = torch.tensor(realAB).to(device).double()

      colorization_model.eval()
      batch_predAB, _ = colorization_model(batchL_3)
      img = cv2.imread(filepath)
      batch_predAB = batch_predAB.cpu().numpy().reshape((224,224,2))
      batchL = batchL.cpu().numpy().reshape((224,224,1))
      realAB = realAB.cpu().numpy().reshape((224,224,2))
      orig = cv2.imread(filepath)
      orig = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), (224,224))
      # orig = reconstruct_no(preprocess(batchL), preprocess(realAB))
      preds = reconstruct_no(preprocess(batchL), preprocess(batch_predAB))
      imag_gird(0, orig, batchL, preds, epoch)
      plt.show()

def create_checkpoint(epoch, netG, optG, netD, optD, max_checkpoint, save_path=config.CHECKPOINT_DIR):
  print('Saving Model and Optimizer weights.....')
  checkpoint = {
        'epoch' : epoch,
        'generator_state_dict' :netG.state_dict(),
        'generator_optimizer': optG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'discriminator_optimizer': optD.state_dict()
    }
  if config.USE_TPU:
    xm.save(checkpoint, f'{save_path}{epoch}_checkpoint.pt')
  else:
    torch.save(checkpoint, f'{save_path}{epoch}_checkpoint.pt')
  print('Weights Saved !!')
  del checkpoint
  files = glob.glob(os.path.expanduser(f"{save_path}*"))
  sorted_files = sorted(files, key=lambda t: -os.stat(t).st_mtime)
  if len(sorted_files) > max_checkpoint:
    os.remove(sorted_files[-1])



def load_checkpoint(checkpoint_directory, netG, optG, netD, optD, device):
    load_from_checkpoint = False
    files = glob.glob(os.path.expanduser(f"{checkpoint_directory}*"))
    for file in files:
        if file.endswith('.pt'):
            load_from_checkpoint=True
            break

    if load_from_checkpoint:
        print('Loading Model and optimizer weights from checkpoint....')
        sorted_files = sorted(files, key=lambda t: -os.stat(t).st_mtime)
        checkpoint = torch.load(f'{sorted_files[0]}')
        epoch_checkpoint = checkpoint['epoch']    
        netG.load_state_dict(checkpoint['generator_state_dict'])
        netG.to(device)

        optG.load_state_dict(checkpoint['generator_optimizer'])

        netD.load_state_dict(checkpoint['discriminator_state_dict'])
        netD.to(device)

        optD.load_state_dict(checkpoint['discriminator_optimizer'])
        print('Loaded Weights !!!')
        

        return netG, optG, netD, optD, epoch_checkpoint
    else:
        print('There are no checkpoints in the mentioned directoy, the Model will train from scratch.')
        epoch_checkpoint = 0
        return netG, optG, netD, optD, epoch
    


def plot_gan_loss(G_losses, D_losses):
  plt.figure(figsize=(10,5))
  plt.title(f"Generator and Discriminator Loss During Training ")
  plt.plot(G_losses,label="G")
  plt.plot(D_losses,label="D")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(f'GANLOSS{epoch}.pdf',figsize=(30,30))
