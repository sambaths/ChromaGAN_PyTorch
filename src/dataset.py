import os
import cv2
import config
import numpy as np


class DATA():
    def __init__(self, dirname, max_len=None):
        self.dir_path = dirname
        self.filelist = os.listdir(self.dir_path)[:max_len]
        self.batch_size = config.BATCH_SIZE
        self.size = len(self.filelist)
        self.data_index = 0
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, item):
        img = []
        label = []
        itemfilelist = ''
        filename = os.path.join(self.dir_path, self.filelist[item])
        itemfilelist = self.filelist[item]
        greyimg, colorimg = self.read_img(filename)
        img = greyimg
        label = colorimg
        img = np.asarray(img)/255 # values between 0 and 1
        label = np.asarray(label)/255 # values between 0 and 1
        return img, label, itemfilelist

    def read_img(self, filename):
        img = cv2.imread(filename, 3)
        height, width, channels = img.shape
        min_hw = int(min(height,width)/2)
        img = img[int(height/2)-min_hw:int(height/2)+min_hw,int(width/2)-min_hw:int(width/2)+min_hw,:]
        labimg = cv2.cvtColor(cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE)), cv2.COLOR_RGB2Lab) ## Changed BGR to RGB
        return np.reshape(labimg[:,:,0], (1, config.IMAGE_SIZE, config.IMAGE_SIZE)), np.reshape(labimg[:, :, 1:], (2,config.IMAGE_SIZE, config.IMAGE_SIZE))

    def generate_batch(self):
        batch = []
        labels = []
        filelist = []
        for i in range(self.batch_size):
            filename = os.path.join(self.dir_path, self.filelist[self.data_index])
            filelist.append(self.filelist[self.data_index])
            greyimg, colorimg = self.read_img(filename)
            batch.append(greyimg)
            labels.append(colorimg)
            self.data_index = (self.data_index + 1) % self.size
        batch = np.asarray(batch)/255 # values between 0 and 1
        labels = np.asarray(labels)/255 # values between 0 and 1
        return batch, labels, filelist

