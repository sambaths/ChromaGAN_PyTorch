
import os
import cv2
import torch
import config
import numpy as np
import matplotlib.pyplot as plt

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

def keep_ckpt(no_of_ckpts_to_keep, MODEL_DIR):
  if len(os.listdir(MODEL_DIR)) > no_of_ckpts_to_keep:
    folder_list = pd.Series(os.listdir(MODEL_DIR)).apply(int).values
    folder_list = np.sort(folder_list)[::-1]
    keep_folder = folder_list[:no_of_ckpts_to_keep+1]
    for value in folder_list:
      if value not in keep_folder:
        print(f'\nRemoving checkpoint from epoch {str(value)}')
        shutil.rmtree(MODEL_DIR+str(value))

def save_ckp(state, MODEL_DIR, fname, intermediate=False):#, best_model_dir):
    f_path = f'{MODEL_DIR}{state["epoch"]}/'
    if intermediate:
      f_path = f'{MODEL_DIR}{state["epoch"]}/checkpoint_{state["batch"]}/'
    try:
      os.makedirs(f'{f_path}', exist_ok=True)
    except:
      pass
    try:  
      print(f'Saving Checkpoint to {f_path}{fname}')
      if use_tpu:
        xm.save(state, f'{f_path}{fname}')
      else:
        torch.save(state, f'{f_path}{fname}')
    except:
      print(f"Couldn't save the model at the specified location. Trying to save to current directory. !!")
      if use_tpu:
        xm.save(state, f'{f_path}{fname}')
      else:
        torch.save(state, f'{fname}')

      print('Saved Model in the current directory.')

def plot_gan_loss(G_losses, D_losses):
  plt.figure(figsize=(10,5))
  plt.title(f"Generator and Discriminator Loss During Training ")
  plt.plot(G_losses,label="G")
  plt.plot(D_losses,label="D")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(f'GANLOSS{epoch}.pdf',figsize=(30,30))
