import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


bias=True

class discriminator_model(nn.Module):

  def __init__(self):
    super(discriminator_model, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=(4,4),padding=1,stride=(2,2),bias=bias) # 64, 112, 112
    self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4), padding=1, stride=(2,2), bias=bias) # 128, 56, 56
    self.conv3 = nn.Conv2d(128,256, kernel_size=(4,4), padding=1, stride=(2,2), bias=bias) # 256, 28, 28, 2
    self.conv4 = nn.Conv2d(256,512, kernel_size=(4,4), padding=3, stride=(1,1), bias=bias) # 512, 28, 28
    self.conv5 = nn.Conv2d(512,1, kernel_size=(4,4), padding=3, stride=(1,1), bias=bias) # 1, 
    self.leaky_relu = nn.LeakyReLU(0.3)

  def forward(self,input):

    net = self.conv1(input)               #[-1, 64, 112, 112]
    net = self.leaky_relu(net)          #[-1, 64, 112, 112]    
    net = self.conv2(net)               #[-1, 128, 56, 56] 
    net = self.leaky_relu(net)          #[-1, 128, 56, 56] 
    net = self.conv3(net)               #[-1, 256, 28, 28]
    net = self.leaky_relu(net)          #[-1, 256, 28, 28]   
    net = self.conv4(net)               #[-1, 512, 27, 27]
    net = self.leaky_relu(net)          #[-1, 512, 27, 27]
    net = self.conv5(net)               #[-1, 1, 26, 26]
    return net

class colorization_model(nn.Module):
  def __init__(self):
    super(colorization_model, self).__init__()

    self.VGG_model = torchvision.models.vgg16(pretrained=True)
    self.VGG_model = nn.Sequential(*list(self.VGG_model.features.children())[:-8]) #[None, 512, 28, 28]
    self.relu = nn.ReLU()
    self.lrelu = nn.LeakyReLU(0.3)
    self.global_features_conv1 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(2,2), bias=bias) #[None, 512, 14, 14]
    self.global_features_bn1 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)
    self.global_features_conv2 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias) #[None, 512, 14, 14]
    self.global_features_bn2 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)
    self.global_features_conv3 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(2,2), bias=bias) #[None, 512, 7, 7]
    self.global_features_bn3 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)
    self.global_features_conv4 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias) #[None, 512, 7, 7]
    self.global_features_bn4 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)

    self.global_features2_flatten = nn.Flatten()
    self.global_features2_dense1 = nn.Linear(512*7*7,1024)
    self.midlevel_conv1 = nn.Conv2d(512,512, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias) #[None, 512, 28, 28]
    self.global_features2_dense2 = nn.Linear(1024,512)
    self.midlevel_bn1 = nn.BatchNorm2d(512, eps=0.001,momentum=0.99)
    self.global_features2_dense3 = nn.Linear(512,256)
    self.midlevel_conv2 = nn.Conv2d(512,256, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)

    self.midlevel_bn2 = nn.BatchNorm2d(256,eps=0.001,momentum=0.99)
   
     #[None, 256, 28, 28]
    # self.midlevel_bn2 = nn.BatchNorm2d(256)#,,eps=0.001,momentum=0.99)

    self.global_featuresClass_flatten = nn.Flatten()
    self.global_featuresClass_dense1 = nn.Linear(512*7*7, 4096)
    self.global_featuresClass_dense2 = nn.Linear(4096, 4096)
    self.global_featuresClass_dense3 = nn.Linear(4096, 1000)
    self.softmax = nn.Softmax()

    self.outputmodel_conv1 = nn.Conv2d(512, 256, kernel_size=(1,1), padding=0, stride=(1,1),  bias=bias) 
    self.outputmodel_conv2 = nn.Conv2d(256, 128, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_conv3 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_conv5 = nn.Conv2d(64, 32, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_conv6 = nn.Conv2d(32, 2, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_upsample = nn.Upsample(scale_factor=(2,2))
    self.outputmodel_bn1 = nn.BatchNorm2d(128)
    self.outputmodel_bn2 = nn.BatchNorm2d(64)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()

  def forward(self,input_img):

    # VGG Without Top Layers

    vgg_out = self.VGG_model(torch.tensor(input_img))

    #Global Features
    
    global_features = self.relu(self.global_features_conv1(vgg_out))  #[None, 512, 14, 14]
    global_features = self.global_features_bn1(global_features) #[None, 512, 14, 14]
    global_features = self.relu(self.global_features_conv2(global_features)) #[None, 512, 14, 14]
    # global_features = self.global_features_bn2(global_features) #[None, 512, 14, 14]

    global_features = self.relu(self.global_features_conv3(global_features)) #[None, 512, 7, 7]
    global_features = self.global_features_bn3(global_features)  #[None, 512, 7, 7]
    global_features = self.relu(self.global_features_conv4(global_features)) #[None, 512, 7, 7]
    # global_features = self.global_features_bn4(global_features) #[None, 512, 7, 7]
    
    global_features2 = self.global_features2_flatten(global_features) #[None, 512*7*7]
    
    global_features2 = self.global_features2_dense1(global_features2) #[None, 1024]
    global_features2 = self.global_features2_dense2(global_features2) #[None, 512]
    global_features2 = self.global_features2_dense3(global_features2) #[None, 256]
    global_features2 = global_features2.unsqueeze(2).expand(-1,256,28*28) #[None, 256, 784]
    global_features2 = global_features2.view((-1,256,28,28)) #[None, 256, 28, 28]

    global_featureClass = self.global_featuresClass_flatten(global_features) #[None, 512*7*7]
    global_featureClass = self.global_featuresClass_dense1(global_featureClass) #[None, 4096]
    global_featureClass = self.global_featuresClass_dense2(global_featureClass) #[None, 4096]
    global_featureClass = self.softmax(self.global_featuresClass_dense3(global_featureClass))#[None, 1000]
    
    # Mid Level Features
    midlevel_features = self.midlevel_conv1(vgg_out) #[None, 512, 28, 28]
    midlevel_features = self.midlevel_bn1(midlevel_features) #[None, 512, 28, 28]
    midlevel_features = self.midlevel_conv2(midlevel_features) #[None, 256, 28, 28]
    midlevel_features = self.midlevel_bn2(midlevel_features) #[None, 256, 28, 28]

    # Fusion of (VGG16 + MidLevel) + (VGG16 + Global)

    modelFusion = torch.cat([midlevel_features, global_features2],dim=1)

    # Fusion Colorization

    outputmodel = self.relu(self.outputmodel_conv1(modelFusion)) # None, 256, 28, 28
    outputmodel = self.relu(self.outputmodel_conv2(outputmodel)) # None, 128, 28, 28

    outputmodel = self.outputmodel_upsample(outputmodel) # None, 128, 56, 56
    outputmodel = self.outputmodel_bn1(outputmodel) # None, 128, 56, 56
    outputmodel = self.relu(self.outputmodel_conv3(outputmodel)) # None, 64, 56, 56
    outputmodel = self.relu(self.outputmodel_conv4(outputmodel)) # None, 64, 56, 56 

    outputmodel = self.outputmodel_upsample(outputmodel) # None, 64, 112, 112
    outputmodel = self.outputmodel_bn2(outputmodel) # None, 64, 112, 112
    outputmodel = self.relu(self.outputmodel_conv5(outputmodel)) # None, 32, 112, 112
    outputmodel = self.sigmoid(self.outputmodel_conv6(outputmodel)) # None, 2, 112, 112
    outputmodel = self.outputmodel_upsample(outputmodel) # None, 2, 224, 224

    return outputmodel, global_featureClass


class GAN(nn.Module):
  def __init__(self, netG, netD):
    super(GAN, self).__init__()

    self.netG = netG
    self.netD = netD

  def forward(self, trainL, trainL_3):

    for param in self.netD.parameters():
      param.requires_grad= False
    
    predAB, classVector = self.netG(trainL_3)
    predLAB = torch.cat([trainL, predAB], dim=1)
    discpred = self.netD(predLAB)

    return predAB, classVector, discpred

