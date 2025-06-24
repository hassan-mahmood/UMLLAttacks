

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from  torch.nn.utils import weight_norm
from utils.utility import * 
import numpy as np 
import random 
from scipy.linalg import subspace_angles

torch.backends.cudnn.deterministic = True
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)
#from torch.nn.utils import prune

class ResNetModel(torch.nn.Module):

	def __init__(self,model_params):
		super(ResNetModel,self).__init__()
		self.num_classes=model_params['num_classes']
		#resnetbasemodel=model_params['base_model'](pretrained=True)
		resnetbasemodel=models.resnet101(pretrained=True)
		self.basemodel = nn.Sequential(*list(resnetbasemodel.children())[:-1])
		self.prelu1=nn.PReLU()
		self.classifier=nn.Linear(in_features=2048,out_features=self.num_classes,bias=True)
		#self.get_model_weights=self.get_common_model_weights
	
	def forward(self,x,params=[]):
		outputfeature=self.basemodel(x)
		globalfeature=self.prelu1(outputfeature)
		globalfeature=torch.flatten(globalfeature,1)
		#globalfeature.requires_grad=True
		output=self.classifier(globalfeature)
		#print('Global feature:',torch.sum(globalfeature),torch.sum(output))
		return output,outputfeature


class netG(nn.Module):
    def __init__(self,  nz):
        super(netG, self).__init__()
        
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(self.nz, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 8, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        # self.tconv6 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
        #     nn.Tanh(),
        # )

        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.tconv7 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.tconv8 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )


    def forward(self, input):
        
        input = input.view(-1, self.nz)
        fc1 = self.fc1(input)
        fc1 = fc1.view(-1, 768, 1, 1)
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        tconv6 = self.tconv6(tconv5)
        tconv7=self.tconv7(tconv6)[:,:,17:241,17:241]
        tconv8=self.tconv8(tconv7)
        return tconv8