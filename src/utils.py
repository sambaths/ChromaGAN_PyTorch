
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

    batchX = batchX.reshape(config.IMAGE_SIZE,config.IMAGE_SIZE,1) 
    predictedY = predictedY.reshape(config.IMAGE_SIZE,config.IMAGE_SIZE,2)
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2RGB)
    save_results_path = os.path.join(config.OUT_DIR,config.TEST_NAME)
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    save_path = os.path.join(save_results_path, filelist +  "_reconstructed.jpg" )
    cv2.imwrite(save_path, result)
    return result



def reconstruct_no(batchX, predictedY):

    batchX = batchX.reshape(config.IMAGE_SIZE,config.IMAGE_SIZE,1) 
    predictedY = predictedY.reshape(config.IMAGE_SIZE,config.IMAGE_SIZE,2)
    
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2RGB)
    return result



def imag_gird(axrow,orig, batchL, preds):
  fig , ax = plt.subplots(1,3, figsize=(15,15))
  ax[0].imshow(orig)
  ax[0].set_title('Original Image')

  ax[1].imshow(np.tile(batchL,(1,1,3)))
  ax[1].set_title('L Image with Channels reapeated(Input)') 

  ax[2].imshow(preds)
  ax[2].set_title('Pred Image')



def plot_some(test_data, colorization_model):
  with torch.no_grad():
  # for batch in range(TOTAL_TEST_BATCH):
    idx = 20 #torch.randint(0, len(test_data), (1,)).item()
    batchL, realAB, filename = test_data[idx]
    filepath = DATA_DIR+TEST_DIR+filename
    batchL = batchL.reshape(1,1,config.IMAGE_SIZE,config.IMAGE_SIZE)
    realAB = realAB.reshape(1,2,config.IMAGE_SIZE,config.IMAGE_SIZE)
    batchL_3 = torch.tensor(np.tile(batchL, [1, 3, 1, 1]))
    batchL_3 = batchL_3.to(DEVICE)
    batchL = torch.tensor(batchL).to(DEVICE).double()
    realAB = torch.tensor(realAB).to(DEVICE).double()

    colorization_model.eval()
    batch_predAB, _ = colorization_model(batchL_3)
    img = cv2.imread(filepath)
    batch_predAB = batch_predAB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
    batchL = batchL.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,1))
    realAB = realAB.cpu().numpy().reshape((config.IMAGE_SIZE,config.IMAGE_SIZE,2))
    orig = cv2.imread(filepath)
    orig = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), (config.IMAGE_SIZE,config.IMAGE_SIZE))
    preds = reconstruct_no(preprocess(batchL), preprocess(batch_predAB))
    imag_gird(0, orig, batchL, preds)
    plt.show()



def save_checkpoint(combined_model, colorization_model, discriminator):
    """Save checkpoint if a new best is achieved"""
    os.makedirs(f'{MODEL_DIR}{epoch}', exist_ok=True)
    save_path = os.path.join(f'{MODEL_DIR}{epoch}', f'combined_epoch{epoch}.pt')
    torch.save(combined_model, save_path)

    save_path = os.path.join(f'{MODEL_DIR}{epoch}', f'colorization_epoch{epoch}.pt')
    torch.save(colorization_model, save_path)

    save_path = os.path.join(f'{MODEL_DIR}{epoch}', f'discriminator_epoch{epoch}.pt')
    torch.save(discriminator, save_path)

    save_path = os.path.join(f'{MODEL_DIR}{epoch}', f'colorization_state_dict_epoch{epoch}.pt')
    torch.save(colorization_model.state_dict(), save_path)

    save_path = os.path.join(f'{MODEL_DIR}{epoch}', f'discriminator_state_dict_epoch{epoch}.pt')
    torch.save(discriminator.state_dict(), save_path)


def keep_ckpt(no_of_ckpts_to_keep, MODEL_DIR):
  if len(os.listdir(MODEL_DIR)) > no_of_ckpts_to_keep:
    folder_list = pd.Series(os.listdir(MODEL_DIR)).apply(int).values
    folder_list = folder_list[-1:].sort()
    keep_folder = folder_list[:no_of_ckpts_to_keep+1]
    for value in folder_list:
      if value not in keep_folder:
        print(f'\nRemoving Models in {value}')
        shutil.rmtree(MODEL_DIR+value)

def save_ckp(state, MODEL_DIR):#, best_model_dir):
    f_path = f'{MODEL_DIR}{state["epoch"]}/'
    os.makedirs(f'{f_path}', exist_ok=True)
    torch.save(state, f'{f_path}checkpoint.pt')
    # if is_best:
    #     best_fpath = best_model_dir / 'best_model.pt'
    #     shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
