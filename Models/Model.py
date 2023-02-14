

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



class MyResNet101(models.resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

# class GUAPModel(torch.nn.Module):
# 	def __init__(self,model_params):
# 		super(GUAPModel,self).__init__()

# 		self.generator=UAPGen()

# 		self.classifier=ResNetModel(model_params)
# 		#checkpoint=torch.load('/mnt/raptor/hassan/AT/weights/voc/common/br_clean/model-696.pt')
# 		checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/res_mean/model-196.pt') 
# 		self.classifier.load_state_dict(checkpoint['model_state'])
# 		self.meanstds = get_pickle_data('/mnt/raptor/hassan/data/KG_data/voc/meanstd')

# 	def forward(self,x,par,targetset=[]):
# 		epsilon_norm=par['epsilon_norm']
		
# 		if(epsilon_norm>0.0):
# 			p_norm,target_label,target_class=par['p_norm'],par['target_label'],par['target_class']	
			
# 			z = torch.FloatTensor(x.shape[0], 120, 1, 1).normal_(0, 1).cuda()
# 			#print(z.shape)
# 			#print(z[:,100,:,:].shape,target_label.shape)
# 			z[:,100:,0,0]=target_label
# 			# out=self.generator(z)
# 			# print('Image:',torch.min(x).item(),torch.max(x).item())
# 			# print('Min and max before',torch.min(out).item(),torch.max(out).item())
# 			# k=normalize_and_scale(out,self.meanstds['means'],self.meanstds['stds'],eps=epsilon_norm)
# 			# print('Min and max after',torch.min(k).item(),torch.max(k).item())

# 			singleuap,classuap=self.generator(z)

# 			for c in range(target_label.shape[1]):
# 				out=classuap[:,c*3:(c+1)*3]
# 				k=normalize_and_scale(out,self.meanstds['means'],self.meanstds['stds'],eps=epsilon_norm)
				

			
# 				recons = x + k
# 				for cii in range(3):
# 					recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(x[:,cii,:,:].min(), x[:,cii,:,:].max())

# 				input_images = recons

# 		else:
# 			input_images = x 

# 		#a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

# 		#k=out*(epsilon_norm/(a+1e-10))[:,None]

# 		#print(torch.min(k),torch.max(k))
# 		#k=out


		
		
# 		# 

# 		####
# 		#k=torch.sign(k)*torch.minimum(torch.abs(k),torch.Tensor([epsilon_norm]).cuda())
# 		#k=torch.sum(k,1)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=2))

# 		#0/0

# 		#print(k.shape)
# 		#0/0

# 		#return self.model(x)
# 		# x: input image
# 		# gt: ground truth label
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
# 		#print('Self u shape:',self.U.shape)
# 		# print('\n',torch.sum(self.U[:,:5,:],dim=2))
# 		#print('\n',torch.sum(self.U,dim=2))
# 		#print('\n',self.U[:,:5,:5])
# 		#k=self.U[target_label[:,target_class].long(),(torch.ones_like(target_label[:,target_class])*target_class).long(),:]
		
# 		#print(k[:5,:5])
# 		#print(k.shape)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=1,keepdim=True))
# 		# print(out.shape)
# 		# 0/0
		
# 		# out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 		# out=torch.sum(out,1)
# 		# a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
# 		# # # #k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
# 		# k=out*(epsilon_norm/(a+1e-10))[:,None]
# 		# # print(torch.min(k),torch.max(k))
# 		# # 0/0

		
		
# 		# k=k.view(k.shape[0],x.shape[1],x.shape[2],-1)
# 		# print(k[:5,0,0,:5])
		
		
# 		#input_images = x
		
# 		input_images=torch.clamp(input_images,min=0.0,max=1.0)
# 		#print('\n',torch.min(k).item(),torch.max(k).item(),torch.min(input_images).item(),torch.max(input_images).item())
# 		#0/0
	
# 		return self.classifier(input_images)



class GUAPModel(torch.nn.Module):
	def __init__(self,model_params):
		super(GUAPModel,self).__init__()

		self.generator=UAPGen()

		self.classifier=ResNetModel(model_params)
		#checkpoint=torch.load('/mnt/raptor/hassan/AT/weights/voc/common/br_clean/model-696.pt')
		checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/res_mean/model-196.pt') 
		self.classifier.load_state_dict(checkpoint['model_state'])
		self.meanstds = get_pickle_data('/mnt/raptor/hassan/data/KG_data/voc/meanstd')

	def forward(self,x,par,targetset=[]):
		epsilon_norm=par['epsilon_norm']
		
		if(epsilon_norm>0.0):
			p_norm,target_label,target_class=par['p_norm'],par['target_label'],par['target_class']	
			
			z = torch.FloatTensor(x.shape[0], 120, 1, 1).normal_(0, 1).cuda()
			#print(z.shape)
			#print(z[:,100,:,:].shape,target_label.shape)
			#x=torch.tile(x,(20,1,1,1))
			z[:,100:,0,0]=target_label
			g_uap,class_uap=self.generator(z)

			g_uap=normalize_and_scale(g_uap,self.meanstds['means'],self.meanstds['stds'],eps=epsilon_norm)
			recons=torch.add(g_uap,x)
			for cii in range(3):
				recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(x[:,cii,:,:].min(), x[:,cii,:,:].max())

			input_images=recons
			g_output,_=self.classifier(input_images)


			x=mtile(x,0,20)
			class_uap=class_uap.view(-1,3,224,224)
			class_uap=normalize_and_scale(class_uap,self.meanstds['means'],self.meanstds['stds'],eps=epsilon_norm)
			
			recons=torch.add(class_uap,x)

			for cii in range(3):
				recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(x[:,cii,:,:].min(), x[:,cii,:,:].max())

			input_images=recons
			c_output,_=self.classifier(input_images)

			# for i in range(9):
			# 	print(class_uap[i,0,0,:4])
			# 0/0
			
			

			#print('Image:',torch.min(x).item(),torch.max(x).item())
			#print('Min and max before',torch.min(out).item(),torch.max(out).item())
			

			#print('Min and max after',torch.min(k).item(),torch.max(k).item())

			#print(out.shape)
			# with torch.no_grad():
			# 	p=torch.clone(out).detach()
			# 	mask=torch.zeros_like(p)
			# 	mask[torch.logical_or(p>epsilon_norm,p<-1*epsilon_norm)]=1.0
			# 	mask = (mask * p)/epsilon_norm + (1 - mask)

			# k = out/mask
			#with torch.no_grad():
			
			##
			#a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			#k=out*(epsilon_norm/(a+1e-10))[:,None]
			##

			#print('Epsilon norm:',epsilon_norm)
			#print(torch.min(k),torch.max(k))


			#input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
			#input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
			return g_output,c_output



		else:
			input_images = x 

		#a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

		#k=out*(epsilon_norm/(a+1e-10))[:,None]

		#print(torch.min(k),torch.max(k))
		#k=out


		
		
		# 

		####
		#k=torch.sign(k)*torch.minimum(torch.abs(k),torch.Tensor([epsilon_norm]).cuda())
		#k=torch.sum(k,1)
		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=2))

		#0/0

		#print(k.shape)
		#0/0

		#return self.model(x)
		# x: input image
		# gt: ground truth label
		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		#print('Self u shape:',self.U.shape)
		# print('\n',torch.sum(self.U[:,:5,:],dim=2))
		#print('\n',torch.sum(self.U,dim=2))
		#print('\n',self.U[:,:5,:5])
		#k=self.U[target_label[:,target_class].long(),(torch.ones_like(target_label[:,target_class])*target_class).long(),:]
		
		#print(k[:5,:5])
		#print(k.shape)
		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=1,keepdim=True))
		# print(out.shape)
		# 0/0
		
		# out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
		# out=torch.sum(out,1)
		# a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
		# # # #k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
		# k=out*(epsilon_norm/(a+1e-10))[:,None]
		# # print(torch.min(k),torch.max(k))
		# # 0/0

		
		
		# k=k.view(k.shape[0],x.shape[1],x.shape[2],-1)
		# print(k[:5,0,0,:5])
		
		
		#input_images = x
		
		#input_images=torch.clamp(input_images,min=0.0,max=1.0)
		#print('\n',torch.min(k).item(),torch.max(k).item(),torch.min(input_images).item(),torch.max(input_images).item())
		#0/0
	
		return self.classifier(input_images)


# class UAPGen(torch.nn.Module):
# 	def __init__(self):
# 		super(UAPGen,self).__init__()
		
# 		self.imageSize=224
# 		self.conv = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(120, 32 * 8, 3, 1, 0, bias=True),
#             #nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=True),
#             nn.BatchNorm2d(32 * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
#             nn.BatchNorm2d(32 * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
#             nn.BatchNorm2d(32 * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
#             nn.BatchNorm2d(32 ),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=True),
#             nn.BatchNorm2d(3 ),
#             nn.ReLU(True),)

# 		self.fc = nn.Sequential(nn.Linear(3*33*33, 512),
# 			nn.BatchNorm1d(512 ),
# 			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
# 			nn.Linear(512, 1024),
# 			nn.BatchNorm1d(1024 ),
# 			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
# 			nn.Linear(1024, 3*self.imageSize*self.imageSize))

# 		self.tanh = nn.Sequential(nn.Tanh(),)


# 	def forward(self,x):
# 		x = self.conv(x)
# 		x = x.view(-1, 3*33*33)
# 		x = self.fc(x)
# 		x = x.view(-1, 3, self.imageSize, self.imageSize)
# 		return x

class UAPGen(torch.nn.Module):
	def __init__(self):
		super(UAPGen,self).__init__()

		
		
		self.imageSize=224
		self.conv = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(120, 32 * 8, 3, 1, 0, bias=True),
			#nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=True),
			nn.BatchNorm2d(32 * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
			#nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
			nn.BatchNorm2d(32 * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
			#nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
			nn.BatchNorm2d(32 * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(32 * 2,     20 * 3 , 3, 2, 1, bias=True),
			#nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
			nn.BatchNorm2d(20 * 3 ),
			nn.ReLU(True),

			nn.ConvTranspose2d(20 * 3,     20 * 3 , 3, 2, 1, bias=True),
			#nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
			nn.BatchNorm2d(20 * 3 ),
			nn.ReLU(True),
			)
		
		self.conv2 = nn.Sequential(
			nn.ConvTranspose2d(20 * 3,     20 * 3 , 3, 1, 1, bias=True),
			#nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
			nn.BatchNorm2d(20*3),
			nn.ReLU(True),           
		)

		self.conv3 = nn.Sequential(
			nn.ConvTranspose2d(20 * 3,     20 , 3, 1, 1, bias=True),
			#nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
			nn.BatchNorm2d(20),
			nn.ReLU(True),

			nn.ConvTranspose2d(20,     3 , 3, 1, 1, bias=True),
			#nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
			nn.BatchNorm2d(3),
			nn.ReLU(True),
		)


		self.fc = nn.Sequential(nn.Linear(3*33*33, 512),
			nn.BatchNorm1d(512 ),
			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
			nn.Linear(512, 1024),
			nn.BatchNorm1d(1024 ),
			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
			nn.Linear(1024, 3*self.imageSize*self.imageSize))

		self.tanh = nn.Sequential(nn.Tanh(),)


	def forward(self,x):
		x = self.conv(x)
		out=x.view(x.shape[0],20,3*33*33)
		out=out.view(-1,3*33*33)
		#print(out.shape)
		out=self.fc(out)
		out=out.view(x.shape[0],20*3,self.imageSize,self.imageSize)
		#print(out.shape)

		conv2_out=self.conv2(out)
		#print(conv2_out.shape)

		conv3_out=self.conv3(conv2_out)
		#print(conv3_out.shape)
		# 0/0
		# x = x.view(-1, 3*33*33)
		# x = self.fc(x)
		# x = x.view(-1, 3, self.imageSize, self.imageSize)
		return conv3_out,conv2_out


# class UAPGen(torch.nn.Module):
# 	def __init__(self):
# 		super(UAPGen,self).__init__()

		
		
# 		self.imageSize=224
# 		self.conv = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(120, 32 * 8, 3, 1, 0, bias=True),
#             #nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=True),
#             nn.BatchNorm2d(32 * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
#             nn.BatchNorm2d(32 * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(32 * 4, 32 * 3, 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
#             nn.BatchNorm2d(32 * 3),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(32 * 3,     32 * 3 , 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
#             nn.BatchNorm2d(32 * 3 ),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(    32 * 3 ,      20 * 3, 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=True),
#             nn.BatchNorm2d(20 * 3 ),
#             nn.ReLU(True),)

# 		self.conv2=nn.Sequential(
# 			nn.ConvTranspose2d(60,60,3,1,1,bias=True),
# 			nn.BatchNorm2d(60 ),
# 			nn.ReLU(True),)
		
# 		self.conv3=nn.Sequential(
# 			nn.ConvTranspose2d(60,20,3,1,1,bias=True),
# 			nn.BatchNorm2d(20 ),
# 			nn.ReLU(True),

# 			nn.ConvTranspose2d(20,3,3,1,1,bias=True),
# 			nn.BatchNorm2d(3 ),
# 			nn.ReLU(True),
# 			)

# 		self.fc = nn.Sequential(nn.Linear(3*33*33, 512),
# 			nn.BatchNorm1d(512 ),
# 			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
# 			nn.Linear(512, 1024),
# 			nn.BatchNorm1d(1024 ),
# 			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
# 			nn.Linear(1024, 3*self.imageSize*self.imageSize))

# 		self.tanh = nn.Sequential(nn.Tanh(),)


# 	def forward(self,x):
# 		x = self.conv(x)
# 		out=x.view(x.shape[0],20,3,33,33)
# 		out=out.view(-1,3*33*33)
# 		out=self.fc(out)
# 		out=out.view(x.shape[0],20*3,self.imageSize,self.imageSize)

# 		print(out.shape)

# 		conv2_out=self.conv2(out)
# 		print(conv2_out.shape)

# 		conv3_out=self.conv3(conv2_out)
# 		print(conv3_out.shape)

# 		# 0/0
# 		# x = x.view(-1, 3*33*33)
# 		# x = self.fc(x)
# 		# x = x.view(-1, 3, self.imageSize, self.imageSize)
# 		return conv3_out,conv2_out

class zGen(torch.nn.Module):
	def __init__(self):
		super(zGen,self).__init__()

		
		
		self.imageSize=224

		self.fc1 = nn.Sequential(nn.Linear(20, 40),
			nn.BatchNorm1d(40 ),
			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
			)
		self.conv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(40, 20 * 3, 3, 1, 0, bias=True),
            #nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=True),
            nn.BatchNorm2d(20 * 3),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(20 * 3, 20, 3, 2, 1, bias=True),
            #nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(20, 10, 3, 2, 1, bias=True),
            #nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(10,     10 , 3, 2, 1, bias=True),
            #nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
            nn.BatchNorm2d(10 ),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    10 ,      3, 3, 2, 1, bias=True),
            #nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=True),
            nn.BatchNorm2d(3 ),
            nn.ReLU(True),)

		self.fc2 = nn.Sequential(nn.Linear(3*33*33, 512),
			nn.BatchNorm1d(512 ),
			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
			nn.Linear(512, 1024),
			nn.BatchNorm1d(1024 ),
			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
			nn.Linear(1024, 3*self.imageSize*self.imageSize))

		self.tanh = nn.Sequential(nn.Tanh(),)


	def forward(self,x):
		
		x=self.fc1(x)

		x=x.view(-1,40,1,1)
		x = self.conv(x)
		x = x.view(-1, 3*33*33)
		
		out=self.fc2(x)

		return out

# class UAPModelCombine(torch.nn.Module):
# 	def __init__(self,model_params):
# 		super(UAPModelCombine,self).__init__()

# 		self.model=ResNetModel(model_params)
# 		self.model.eval()

# 		self.zgenmodel=zGen()
# 		self.zgenmodel.requires_grad=True

# 		checkpoint=torch.load('/mnt/raptor/hassan/AT/weights/voc/common/br_clean/model-696.pt')
# 		#checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/res_mean/model-196.pt') 
# 		#checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/br_l2_1/model-13.pt')
# 		self.model.load_state_dict(checkpoint['model_state'])
		
# 		#self.model.load_state_dict(checkpoint['model_state'])

		
# 		self.num_classes=model_params['num_classes']
		
# 		# freeze this model

# 		for p in self.model.parameters():
# 			p.requires_grad=False

# 		for p in self.zgenmodel.parameters():
# 			p.requires_grad=True

# 		# first row of U would represent all class UAPs with target label 0 and 
# 		# second row would be for all class UAPs of target label 1
# 		#self.U = torch.nn.Parameter(60*torch.rand((2,self.num_classes,3*224*224)).cuda())
# 		self.U = torch.nn.Parameter(torch.zeros((2,self.num_classes,3*224*224),dtype=torch.float32).cuda())
# 		self.U.requires_grad=True
# 		#self.U.register_hook(lambda grad: torch.sign(grad) * 0.1)
		
# 		#self.uap_weights = torch.nn.Parameter(torch.ones((2,self.num_classes),dtype=torch.float32).cuda())
# 		#self.uap_weights = torch.nn.Parameter(torch.rand((2,self.num_classes)).cuda())
# 		#self.uap_weights.requires_grad=True
# 		#self.Normalize_UAP(p_norm=2)

# 		#self.U=torch.nn.utils.weight_norm(U,name='weight',dim=2)
# 		#U_Module = torch.utils.weight_norm(self.U, name='weight',dim=2)

# 		self.CombineNet = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.Conv2d(63, 20*3, 3, 1, 1,1, bias=True),
#             #nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=True),
#             nn.BatchNorm2d(20 * 3),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.Conv2d(20 * 3, 20, 3, 1, 1,1, bias=True),
#             #nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
#             nn.BatchNorm2d(20),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.Conv2d(20, 10, 3, 1, 1,1, bias=True),
#             #nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
#             nn.BatchNorm2d(10),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.Conv2d(10,     10 , 3, 1, 1, 1,bias=True),
#             #nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
#             nn.BatchNorm2d(10 ),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.Conv2d(    10 ,      3, 3, 1, 1, 1, bias=True),
#             nn.Tanh(),
#             #nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=True),
#             )
# 		self.CombineNet.requires_grad=True

	
# 	def get_norms(self):
# 		#return torch.norm(self.U, dim=2, keepdim=True)
# 		return self.U
# 		#return torch.sum(self.U)
# 		return torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)
# 		#return torch.linalg.vector_norm(self.U,ord=2,dim=2,keepdim=True)
	
# 	def dotemp(self):
# 		# self.U = torch.nn.Parameter(torch.zeros((2,self.num_classes,3*224*224),dtype=torch.float32).cuda())
# 		# self.U.requires_grad=True
# 		#self.U.register_hook(lambda grad: torch.sign(grad) * 0.1)
		
# 		#self.uap_weights = torch.nn.Parameter(torch.ones((2,self.num_classes),dtype=torch.float32).cuda())
# 		#self.uap_weights.requires_grad=True
# 		return 

# 	#def get_UAPs(self):
# 	def get_params(self):

# 		return self.U
# 		#return self.U


# 	def get_UAPs_features(self):

# 		#k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
# 		#Uaps=U
# 		#k=rescale_to_image_range(self.U,dimval=2)
# 		#k=rescale_to_image_range(self.U*self.uap_weights[:,:,None],dimval=2)
# 		k=rescale_to_image_range(self.U,dimval=2)

# 		# with torch.no_grad():
# 		# 	normval=torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10
# 		# 	#k=self.U/normval
# 		# # k=torch.clamp(k,min=0.0,max=1.0)
# 		# k=self.U/normval
# 		# print('min:',torch.min(k),torch.max(k))
		

# 		all_features=[]
# 		all_outputs=[]
# 		for j in range(2):
# 			tempk=k[j,:,:]
# 			outputs,features=self.model(tempk.view(tempk.shape[0],3,224,224))
# 			all_outputs.append(outputs)
# 			all_features.append(features)
# 			# tempk=k[j,:,:]+1
# 			# k_norm=torch.linalg.vector_norm(tempk,ord=float('inf'),dim=1)
# 			# tempk=torch.clamp(tempk/(k_norm+1e-10)[:,None],min=0.0,max=1.0)
# 			# _,features=self.model(tempk.view(tempk.shape[0],3,224,224))
		
		
# 		return all_outputs,all_features
# 		#return all_features[0],all_features[1]
# 		#return features.view(2,self.num_classes,-1)

# 	def Normalize_UAP(self,p_norm,eps_norm):
# 		#print(self.U.grad[:,:5,:5])
		
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
# 		#print(self.U[:,0,:6])
		
# 		with torch.no_grad():
# 			#self.U.div_(torch.norm(self.U, dim=float('inf'), keepdim=True))
# 			#self.U.div_(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10)

# 			if(p_norm==float('inf')):
# 				print('normalizing at:',eps_norm)
# 				0/0
# 				self.U.data=normalize_vec(self.U,max_norm=eps_norm,norm_p=p_norm)
			
# 			else:
# 				a=torch.linalg.vector_norm(self.U,ord=p_norm,dim=2)
# 				#t=torch.minimum(torch.Tensor([eps_norm]).cuda(),a)
# 				#self.U.data=self.U*(t/(a+1e-30))[:,:,None]
# 				self.U.data=self.U*(eps_norm/(a+1e-10))[:,:,None]

# 			#print(torch.linalg.vector_norm(self.U,ord=p_norm,dim=2))
			
# 			#self.U.data=self.U*(torch.min(eps_norm,a+1e-10)/(a+1e-10))[:,:,None]
# 			#print(k.shape)
# 			#print(torch.linalg.vector_norm(k,ord=p_norm,dim=2))

# 			#0/0
# 			#self.U.data=normalize_vec(self.U.view(-1,3,224,224),max_norm=1.0,norm_p=float(2))
# 			#self.uap_weights.data=self.uap_weights/torch.sum(self.uap_weights,dim=1)[:,None]
			

# 		#self.U = F.normalize(self.U, p=2.0, dim=2)
# 		#torch.nn.utils.weight_norm(self.U,name='weight',dim=2)
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		

# 	def forward(self,x,par,targetset=[]):
# 		epsilon_norm=par['epsilon_norm']
		
# 		#k=epsilon_norm*self.U[target_label[:,target_class].long(),target_class,:]
# 		# k=self.U[target_label[:,target_class].long(),target_class,:]
		

# 		# print(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2))
# 		# 0/0
		
# 		#print(out.shape)
# 		if(epsilon_norm>0.0):
# 			p_norm,target_label,target_class,combine_uaps=par['p_norm'],par['target_label'],par['target_class'],par['combine_uaps']
# 			if(combine_uaps):
# 				out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
				
# 				conditional=self.zgenmodel(target_label)
				
# 				conditional = conditional.view(conditional.shape[0],1,-1)
				
# 				out = torch.cat((out,conditional),dim=1)

				
# 				out=out.view(out.shape[0],out.shape[1]*3,224,224)
# 				out=self.CombineNet(out)
# 				out = out.view(out.shape[0],1,-1)
				

# 				#print(torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],1).shape)
# 				##mul_factor=self.uap_weights[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
# 				##out=out*mul_factor[:,:,None]
# 			else:
# 				out=self.U[target_label[:,target_class].long(),target_class,:]
				
# 				##mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
# 				##out=out*mul_factor[:,None]
# 				#out=self.U[target_label[:,target_class].long(),target_class,:]
# 				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
# 				#total=out*mul_factor[:,None]
				
# 				# total=out
# 				# #print(total.shape)
# 				# for ts in targetset[1:]:
# 				# 	ts=torch.Tensor([ts]).long()
# 				# 	out=self.U[target_label[:,ts].long(),ts,:]
# 				# 	#mul_factor=self.uap_weights[target_label[:,ts].long(),ts]
# 				# 	#print(out.shape,mul_factor.shape,(out*mul_factor[:,None]).shape)
# 				# 	#total+=out*mul_factor[:,None]
# 				# 	total+=out

# 				# out = total 
# 			#print(out.shape)
# 			#0/0
# 			#out=out.squeeze()
# 			#print(out.shape)
# 			# print(out[:5,:10])
# 			out=torch.sum(out,1)
			
			
# 			# with torch.no_grad():
# 			# 	p=torch.clone(out).detach()
# 			# 	mask=torch.zeros_like(p)
# 			# 	mask[torch.logical_or(p>epsilon_norm,p<-1*epsilon_norm)]=1.0
# 			# 	mask = (mask * p)/epsilon_norm + (1 - mask)

# 			# k = out/mask
# 			#with torch.no_grad():

# 			# a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
# 			# k=out*(epsilon_norm/(a+1e-10))[:,None]
# 			# print('out min max:',torch.min(k),torch.max(k))

			
# 			a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
# 			t=torch.minimum(torch.Tensor([epsilon_norm]).cuda(),a)
# 			k=out*(t/(a+1e-30))[:,None]
			

# 			#k=out

# 			#a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
# 			#anew=torch.linalg.vector_norm(k,ord=p_norm,dim=1)
# 			#print('Epsilon norm:',epsilon_norm)
# 			#print('Vector norm:',a[0],anew[0],'Min Max:',torch.min(k),torch.max(k))


# 			#input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
# 			input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)

# 		else:
# 			input_images = x 

# 		#a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

# 		#k=out*(epsilon_norm/(a+1e-10))[:,None]

# 		#print(torch.min(k),torch.max(k))
# 		#k=out


		
		
# 		# 

# 		####
# 		#k=torch.sign(k)*torch.minimum(torch.abs(k),torch.Tensor([epsilon_norm]).cuda())
# 		#k=torch.sum(k,1)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=2))

# 		#0/0

# 		#print(k.shape)
# 		#0/0

# 		#return self.model(x)
# 		# x: input image
# 		# gt: ground truth label
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
# 		#print('Self u shape:',self.U.shape)
# 		# print('\n',torch.sum(self.U[:,:5,:],dim=2))
# 		#print('\n',torch.sum(self.U,dim=2))
# 		#print('\n',self.U[:,:5,:5])
# 		#k=self.U[target_label[:,target_class].long(),(torch.ones_like(target_label[:,target_class])*target_class).long(),:]
		
# 		#print(k[:5,:5])
# 		#print(k.shape)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=1,keepdim=True))
# 		# print(out.shape)
# 		# 0/0
		
# 		# out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 		# out=torch.sum(out,1)
# 		# a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
# 		# # # #k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
# 		# k=out*(epsilon_norm/(a+1e-10))[:,None]
# 		# # print(torch.min(k),torch.max(k))
# 		# # 0/0

		
		
# 		# k=k.view(k.shape[0],x.shape[1],x.shape[2],-1)
# 		# print(k[:5,0,0,:5])
		
		
# 		#input_images = x
		
# 		input_images=torch.clamp(input_images,min=0.0,max=1.0)
# 		#print('\n',torch.min(k).item(),torch.max(k).item(),torch.min(input_images).item(),torch.max(input_images).item())
# 		#0/0
# 		#out,features=self.model(input_images)
# 		#return input_images.detach(),out,features
# 		return self.model(input_images)

# 		# print(torch.linalg.vector_norm(k,ord=2,dim=1))

class UAPModelCombine(torch.nn.Module):
	def __init__(self,model_params):
		super(UAPModelCombine,self).__init__()

		self.model=ResNetModel(model_params)
		self.model.eval()

		checkpoint=torch.load('/mnt/raptor/hassan/AT/weights/voc/common/br_clean/model-696.pt')
		#checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/res_mean/model-196.pt') 
		#checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/br_l2_1/model-13.pt')
		self.model.load_state_dict(checkpoint['model_state'])
		#self.model.load_state_dict(checkpoint['model_state'])
		self.num_classes=model_params['num_classes']
		# freeze this model

		for p in self.model.parameters():
			p.requires_grad=False

		# first row of U would represent all class UAPs with target label 0 and 
		# second row would be for all class UAPs of target label 1
		#self.U = torch.nn.Parameter(60*torch.rand((2,self.num_classes,3*224*224)).cuda())
		self.U = torch.nn.Parameter(torch.zeros((2,self.num_classes,3*224*224),dtype=torch.float32).cuda())
		#self.U = torch.nn.Parameter(torch.zeros((2*self.num_classes,3*224*224),dtype=torch.float32).cuda())
		self.U.requires_grad=True
		#self.U.register_hook(lambda grad: torch.sign(grad) * 0.1)
		
		#self.uap_weights = torch.nn.Parameter((1/(self.num_classes*2))*torch.ones((2,self.num_classes,2*self.num_classes),dtype=torch.float32).cuda())
		#self.uap_weights = torch.nn.Parameter(torch.rand((2,self.num_classes)).cuda())
		#self.uap_weights.requires_grad=True
		#self.Normalize_UAP(p_norm=2)

		#self.U=torch.nn.utils.weight_norm(U,name='weight',dim=2)
		#U_Module = torch.utils.weight_norm(self.U, name='weight',dim=2)

		# self.CombineNet = nn.Sequential(
        #     # input is Z, going into a convolution
        #     nn.Conv2d(63, 20*3, 3, 1, 1,1, bias=True),
        #     #nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=True),
        #     nn.BatchNorm2d(20 * 3),
        #     nn.ReLU(True),
        #     # state size. (ngf*8) x 4 x 4
        #     nn.Conv2d(20 * 3, 20, 3, 1, 1,1, bias=True),
        #     #nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
        #     nn.BatchNorm2d(20),
        #     nn.ReLU(True),
        #     # state size. (ngf*4) x 8 x 8
        #     nn.Conv2d(20, 10, 3, 1, 1,1, bias=True),
        #     #nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
        #     nn.BatchNorm2d(10),
        #     nn.ReLU(True),
        #     # state size. (ngf*2) x 16 x 16
        #     nn.Conv2d(10,     10 , 3, 1, 1, 1,bias=True),
        #     #nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
        #     nn.BatchNorm2d(10 ),
        #     nn.ReLU(True),
        #     # state size. (ngf) x 32 x 32
        #     nn.Conv2d(    10 ,      3, 3, 1, 1, 1, bias=True),
        #     nn.Tanh(),
        #     #nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=True),
        #     )
		# self.CombineNet.requires_grad=True

	
	def get_norms(self):
		#return torch.norm(self.U, dim=2, keepdim=True)
		return self.U
		#return torch.sum(self.U)
		return torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)
		#return torch.linalg.vector_norm(self.U,ord=2,dim=2,keepdim=True)
	
	def dotemp(self):
		# self.U = torch.nn.Parameter(torch.zeros((2,self.num_classes,3*224*224),dtype=torch.float32).cuda())
		# self.U.requires_grad=True
		#self.U.register_hook(lambda grad: torch.sign(grad) * 0.1)
		
		#self.uap_weights = torch.nn.Parameter(torch.ones((2,self.num_classes),dtype=torch.float32).cuda())
		#self.uap_weights.requires_grad=True
		return 

	#def get_UAPs(self):
	def get_params(self):

		return self.U
		#return self.U


	def get_UAPs_features(self):

		#k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
		#Uaps=U
		#k=rescale_to_image_range(self.U,dimval=2)
		#k=rescale_to_image_range(self.U*self.uap_weights[:,:,None],dimval=2)
		k=rescale_to_image_range(self.U,dimval=2)

		# with torch.no_grad():
		# 	normval=torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10
		# 	#k=self.U/normval
		# # k=torch.clamp(k,min=0.0,max=1.0)
		# k=self.U/normval
		# print('min:',torch.min(k),torch.max(k))
		

		all_features=[]
		all_outputs=[]
		for j in range(2):
			tempk=k[j,:,:]
			outputs,features=self.model(tempk.view(tempk.shape[0],3,224,224))
			all_outputs.append(outputs)
			all_features.append(features)
			# tempk=k[j,:,:]+1
			# k_norm=torch.linalg.vector_norm(tempk,ord=float('inf'),dim=1)
			# tempk=torch.clamp(tempk/(k_norm+1e-10)[:,None],min=0.0,max=1.0)
			# _,features=self.model(tempk.view(tempk.shape[0],3,224,224))
		
		
		return all_outputs,all_features
		#return all_features[0],all_features[1]
		#return features.view(2,self.num_classes,-1)

	def Normalize_UAP(self,p_norm,eps_norm):
		#print(self.U.grad[:,:5,:5])
		
		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		#print(self.U[:,0,:6])
		
		with torch.no_grad():
			#self.U.div_(torch.norm(self.U, dim=float('inf'), keepdim=True))
			#self.U.div_(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10)

			if(p_norm==float('inf')):
				self.U.data=normalize_vec(self.U,max_norm=eps_norm,norm_p=p_norm)
				#self.uap_weights.data=torch.nn.functional.normalize(self.uap_weights.data,p=1.0,dim=1)[:,None]
				#torch.nn.functional.normalize(self.uap_weights.data,)
				#self.U.weights.div_(torch.linalg.vector_norm())
				#self.uap_weights.div_(torch.sum(self.uap_weights,dim=1)[:,None])
				#self.uap_weights.data=self.uap_weights/torch.sum(self.uap_weights,dim=1)[:,None]
			
			else:
				0/0
				a=torch.linalg.vector_norm(self.U,ord=p_norm,dim=2)
				#t=torch.minimum(torch.Tensor([eps_norm]).cuda(),a)
				#self.U.data=self.U*(t/(a+1e-30))[:,:,None]
				self.U.data=self.U*(eps_norm/(a+1e-10))[:,:,None]

			#print(torch.linalg.vector_norm(self.U,ord=p_norm,dim=2))
			
			#self.U.data=self.U*(torch.min(eps_norm,a+1e-10)/(a+1e-10))[:,:,None]
			#print(k.shape)
			#print(torch.linalg.vector_norm(k,ord=p_norm,dim=2))

			#0/0
			#self.U.data=normalize_vec(self.U.view(-1,3,224,224),max_norm=1.0,norm_p=float(2))
			#self.uap_weights.data=self.uap_weights/torch.sum(self.uap_weights,dim=1)[:,None]
			

		#self.U = F.normalize(self.U, p=2.0, dim=2)
		#torch.nn.utils.weight_norm(self.U,name='weight',dim=2)
		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		

	def forward(self,x,par,targetset=[]):
		epsilon_norm=par['epsilon_norm']
		
		#k=epsilon_norm*self.U[target_label[:,target_class].long(),target_class,:]
		# k=self.U[target_label[:,target_class].long(),target_class,:]
		

		# print(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2))
		# 0/0
		
		#print(out.shape)
		if(epsilon_norm>0.0):
			p_norm,target_label,target_class,combine_uaps=par['p_norm'],par['target_label'],par['target_class'],par['combine_uaps']
			if(combine_uaps):
				#target_class=torch.arange(target_label.shape[1])
				out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
				

				#batch_w=self.uap_weights[target_label[:,target_class].long(),target_class,:].squeeze()
				#out=batch_w[:,:,None]*self.U[None,:,:]

				#out=self.uap_weights[target_label[:,target_class],target_class,:].squeeze()
				#print(self.U.shape)
				#print(out.shape)
				#out=out[:,:,None]*self.U.expand(target_label.shape[0],-1,-1).long()
				#print('Self u shape:',self.U.shape)
				#print(out.shape)
				#a=self.U[:,torch.range(0,self.num_classes*2-1).long().expand(target_label.shape[0]).long(),:]
				#print(a.shape)
				#0/0
				#out=out[:,:,None]*self.U[torch.range(0,self.num_classes*2-1).long().expand(target_label.shape[0]).long(),:]
				
				#out=self.U*self.uap_weights[target_label[:,target_class].long,target_class,:]
				
				# conditional=self.zgenmodel(target_label)
				
				# conditional = conditional.view(conditional.shape[0],1,-1)
				
				# out = torch.cat((out,conditional),dim=1)

				
				# out=out.view(out.shape[0],out.shape[1]*3,224,224)
				# out=self.CombineNet(out)
				# out = out.view(out.shape[0],1,-1)
				

				#print(torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],1).shape)
				##mul_factor=self.uap_weights[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
				##out=out*mul_factor[:,:,None]
			else:
				0/0
				out=self.U[target_label[:,target_class].long(),target_class,:]
				
				##mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
				##out=out*mul_factor[:,None]
				#out=self.U[target_label[:,target_class].long(),target_class,:]
				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
				#total=out*mul_factor[:,None]
				
				# total=out
				# #print(total.shape)
				# for ts in targetset[1:]:
				# 	ts=torch.Tensor([ts]).long()
				# 	out=self.U[target_label[:,ts].long(),ts,:]
				# 	#mul_factor=self.uap_weights[target_label[:,ts].long(),ts]
				# 	#print(out.shape,mul_factor.shape,(out*mul_factor[:,None]).shape)
				# 	#total+=out*mul_factor[:,None]
				# 	total+=out

				# out = total 
			#print(out.shape)
			#0/0
			#out=out.squeeze()
			#print(out.shape)
			# print(out[:5,:10])
			out=torch.sum(out,1).squeeze()
			
			
			# with torch.no_grad():
			# 	p=torch.clone(out).detach()
			# 	mask=torch.zeros_like(p)
			# 	mask[torch.logical_or(p>epsilon_norm,p<-1*epsilon_norm)]=1.0
			# 	mask = (mask * p)/epsilon_norm + (1 - mask)

			# k = out/mask
			#with torch.no_grad():

			# a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
			# k=out*(epsilon_norm/(a+1e-10))[:,None]
			# print('out min max:',torch.min(k),torch.max(k))


			# a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			# #t=torch.minimum(torch.Tensor([epsilon_norm]).cuda(),a)
			# k=out*(epsilon_norm/(a+1e-30))[:,None]

			a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

			k=out*(epsilon_norm/(a+1e-10))[:,None]
			#k=out 
			#print(torch.min(k),torch.max(k))
			

			#k=out

			#a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
			#anew=torch.linalg.vector_norm(k,ord=p_norm,dim=1)
			#print('Epsilon norm:',epsilon_norm)
			#print('Vector norm:',a[0],anew[0],'Min Max:',torch.min(k),torch.max(k))


			#input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
			input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)

		else:
			input_images = x 

		#a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

		#k=out*(epsilon_norm/(a+1e-10))[:,None]

		#print(torch.min(k),torch.max(k))
		#k=out


		
		
		# 

		####
		#k=torch.sign(k)*torch.minimum(torch.abs(k),torch.Tensor([epsilon_norm]).cuda())
		#k=torch.sum(k,1)
		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=2))

		#0/0

		#print(k.shape)
		#0/0

		#return self.model(x)
		# x: input image
		# gt: ground truth label
		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		#print('Self u shape:',self.U.shape)
		# print('\n',torch.sum(self.U[:,:5,:],dim=2))
		#print('\n',torch.sum(self.U,dim=2))
		#print('\n',self.U[:,:5,:5])
		#k=self.U[target_label[:,target_class].long(),(torch.ones_like(target_label[:,target_class])*target_class).long(),:]
		
		#print(k[:5,:5])
		#print(k.shape)
		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=1,keepdim=True))
		# print(out.shape)
		# 0/0
		
		# out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
		# out=torch.sum(out,1)
		# a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
		# # # #k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
		# k=out*(epsilon_norm/(a+1e-10))[:,None]
		# # print(torch.min(k),torch.max(k))
		# # 0/0

		
		
		# k=k.view(k.shape[0],x.shape[1],x.shape[2],-1)
		# print(k[:5,0,0,:5])
		
		
		#input_images = x
		
		input_images=torch.clamp(input_images,min=0.0,max=1.0)
		#print('\n',torch.min(k).item(),torch.max(k).item(),torch.min(input_images).item(),torch.max(input_images).item())
		#0/0
		#out,features=self.model(input_images)
		#return input_images.detach(),out,features
		return self.model(input_images)

		# print(torch.linalg.vector_norm(k,ord=2,dim=1))

# class UAPModelCombine_Weighted(torch.nn.Module):
# 	def __init__(self,model_params):
# 		super(UAPModelCombine,self).__init__()

# 		self.model=ResNetModel(model_params)
# 		self.model.eval()

# 		checkpoint=torch.load('/mnt/raptor/hassan/AT/weights/voc/common/br_clean/model-696.pt')
# 		#checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/res_mean/model-196.pt') 
# 		#checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/br_l2_1/model-13.pt')
# 		self.model.load_state_dict(checkpoint['model_state'])
# 		#self.model.load_state_dict(checkpoint['model_state'])
# 		self.num_classes=model_params['num_classes']
# 		# freeze this model

# 		for p in self.model.parameters():
# 			p.requires_grad=False

# 		# first row of U would represent all class UAPs with target label 0 and 
# 		# second row would be for all class UAPs of target label 1
# 		#self.U = torch.nn.Parameter(60*torch.rand((2,self.num_classes,3*224*224)).cuda())
# 		#self.U = torch.nn.Parameter(torch.zeros((2,self.num_classes,3*224*224),dtype=torch.float32).cuda())
# 		self.U = torch.nn.Parameter(torch.zeros((2*self.num_classes,3*224*224),dtype=torch.float32).cuda())
# 		self.U.requires_grad=True
# 		#self.U.register_hook(lambda grad: torch.sign(grad) * 0.1)
		
# 		self.uap_weights = torch.nn.Parameter((1/(self.num_classes*2))*torch.ones((2,self.num_classes,2*self.num_classes),dtype=torch.float32).cuda())
# 		#self.uap_weights = torch.nn.Parameter(torch.rand((2,self.num_classes)).cuda())
# 		self.uap_weights.requires_grad=True
# 		#self.Normalize_UAP(p_norm=2)

# 		#self.U=torch.nn.utils.weight_norm(U,name='weight',dim=2)
# 		#U_Module = torch.utils.weight_norm(self.U, name='weight',dim=2)

# 		# self.CombineNet = nn.Sequential(
#         #     # input is Z, going into a convolution
#         #     nn.Conv2d(63, 20*3, 3, 1, 1,1, bias=True),
#         #     #nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=True),
#         #     nn.BatchNorm2d(20 * 3),
#         #     nn.ReLU(True),
#         #     # state size. (ngf*8) x 4 x 4
#         #     nn.Conv2d(20 * 3, 20, 3, 1, 1,1, bias=True),
#         #     #nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
#         #     nn.BatchNorm2d(20),
#         #     nn.ReLU(True),
#         #     # state size. (ngf*4) x 8 x 8
#         #     nn.Conv2d(20, 10, 3, 1, 1,1, bias=True),
#         #     #nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
#         #     nn.BatchNorm2d(10),
#         #     nn.ReLU(True),
#         #     # state size. (ngf*2) x 16 x 16
#         #     nn.Conv2d(10,     10 , 3, 1, 1, 1,bias=True),
#         #     #nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
#         #     nn.BatchNorm2d(10 ),
#         #     nn.ReLU(True),
#         #     # state size. (ngf) x 32 x 32
#         #     nn.Conv2d(    10 ,      3, 3, 1, 1, 1, bias=True),
#         #     nn.Tanh(),
#         #     #nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=True),
#         #     )
# 		# self.CombineNet.requires_grad=True

	
# 	def get_norms(self):
# 		#return torch.norm(self.U, dim=2, keepdim=True)
# 		return self.U
# 		#return torch.sum(self.U)
# 		return torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)
# 		#return torch.linalg.vector_norm(self.U,ord=2,dim=2,keepdim=True)
	
# 	def dotemp(self):
# 		# self.U = torch.nn.Parameter(torch.zeros((2,self.num_classes,3*224*224),dtype=torch.float32).cuda())
# 		# self.U.requires_grad=True
# 		#self.U.register_hook(lambda grad: torch.sign(grad) * 0.1)
		
# 		#self.uap_weights = torch.nn.Parameter(torch.ones((2,self.num_classes),dtype=torch.float32).cuda())
# 		#self.uap_weights.requires_grad=True
# 		return 

# 	#def get_UAPs(self):
# 	def get_params(self):

# 		return self.U
# 		#return self.U


# 	def get_UAPs_features(self):

# 		#k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
# 		#Uaps=U
# 		#k=rescale_to_image_range(self.U,dimval=2)
# 		#k=rescale_to_image_range(self.U*self.uap_weights[:,:,None],dimval=2)
# 		k=rescale_to_image_range(self.U,dimval=2)

# 		# with torch.no_grad():
# 		# 	normval=torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10
# 		# 	#k=self.U/normval
# 		# # k=torch.clamp(k,min=0.0,max=1.0)
# 		# k=self.U/normval
# 		# print('min:',torch.min(k),torch.max(k))
		

# 		all_features=[]
# 		all_outputs=[]
# 		for j in range(2):
# 			tempk=k[j,:,:]
# 			outputs,features=self.model(tempk.view(tempk.shape[0],3,224,224))
# 			all_outputs.append(outputs)
# 			all_features.append(features)
# 			# tempk=k[j,:,:]+1
# 			# k_norm=torch.linalg.vector_norm(tempk,ord=float('inf'),dim=1)
# 			# tempk=torch.clamp(tempk/(k_norm+1e-10)[:,None],min=0.0,max=1.0)
# 			# _,features=self.model(tempk.view(tempk.shape[0],3,224,224))
		
		
# 		return all_outputs,all_features
# 		#return all_features[0],all_features[1]
# 		#return features.view(2,self.num_classes,-1)

# 	def Normalize_UAP(self,p_norm,eps_norm):
# 		#print(self.U.grad[:,:5,:5])
		
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
# 		#print(self.U[:,0,:6])
		
# 		with torch.no_grad():
# 			#self.U.div_(torch.norm(self.U, dim=float('inf'), keepdim=True))
# 			#self.U.div_(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10)

# 			if(p_norm==float('inf')):
# 				self.U.data=normalize_vec(self.U,max_norm=eps_norm,norm_p=p_norm)
# 				#self.uap_weights.data=torch.nn.functional.normalize(self.uap_weights.data,p=1.0,dim=1)[:,None]
# 				#torch.nn.functional.normalize(self.uap_weights.data,)
# 				#self.U.weights.div_(torch.linalg.vector_norm())
# 				#self.uap_weights.div_(torch.sum(self.uap_weights,dim=1)[:,None])
# 				#self.uap_weights.data=self.uap_weights/torch.sum(self.uap_weights,dim=1)[:,None]
			
# 			else:
# 				0/0
# 				a=torch.linalg.vector_norm(self.U,ord=p_norm,dim=2)
# 				#t=torch.minimum(torch.Tensor([eps_norm]).cuda(),a)
# 				#self.U.data=self.U*(t/(a+1e-30))[:,:,None]
# 				self.U.data=self.U*(eps_norm/(a+1e-10))[:,:,None]

# 			#print(torch.linalg.vector_norm(self.U,ord=p_norm,dim=2))
			
# 			#self.U.data=self.U*(torch.min(eps_norm,a+1e-10)/(a+1e-10))[:,:,None]
# 			#print(k.shape)
# 			#print(torch.linalg.vector_norm(k,ord=p_norm,dim=2))

# 			#0/0
# 			#self.U.data=normalize_vec(self.U.view(-1,3,224,224),max_norm=1.0,norm_p=float(2))
# 			#self.uap_weights.data=self.uap_weights/torch.sum(self.uap_weights,dim=1)[:,None]
			

# 		#self.U = F.normalize(self.U, p=2.0, dim=2)
# 		#torch.nn.utils.weight_norm(self.U,name='weight',dim=2)
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		

# 	def forward(self,x,par,targetset=[]):
# 		epsilon_norm=par['epsilon_norm']
		
# 		#k=epsilon_norm*self.U[target_label[:,target_class].long(),target_class,:]
# 		# k=self.U[target_label[:,target_class].long(),target_class,:]
		

# 		# print(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2))
# 		# 0/0
		
# 		#print(out.shape)
# 		if(epsilon_norm>0.0):
# 			p_norm,target_label,target_class,combine_uaps=par['p_norm'],par['target_label'],par['target_class'],par['combine_uaps']
# 			if(combine_uaps):
# 				#target_class=torch.arange(target_label.shape[1])
# 				# out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 				batch_w=self.uap_weights[target_label[:,target_class].long(),target_class,:].squeeze()
# 				#out=self.uap_weights[target_label[:,target_class],target_class,:].squeeze()
# 				#print(self.U.shape)
# 				#print(out.shape)
# 				#out=out[:,:,None]*self.U.expand(target_label.shape[0],-1,-1).long()
# 				#print('Self u shape:',self.U.shape)
# 				#print(out.shape)
# 				#a=self.U[:,torch.range(0,self.num_classes*2-1).long().expand(target_label.shape[0]).long(),:]
# 				#print(a.shape)
# 				#0/0
# 				#out=out[:,:,None]*self.U[torch.range(0,self.num_classes*2-1).long().expand(target_label.shape[0]).long(),:]
# 				out=batch_w[:,:,None]*self.U[None,:,:]

				
# 				#out=self.U*self.uap_weights[target_label[:,target_class].long,target_class,:]
				
# 				# conditional=self.zgenmodel(target_label)
				
# 				# conditional = conditional.view(conditional.shape[0],1,-1)
				
# 				# out = torch.cat((out,conditional),dim=1)

				
# 				# out=out.view(out.shape[0],out.shape[1]*3,224,224)
# 				# out=self.CombineNet(out)
# 				# out = out.view(out.shape[0],1,-1)
				

# 				#print(torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],1).shape)
# 				##mul_factor=self.uap_weights[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
# 				##out=out*mul_factor[:,:,None]
# 			else:
# 				0/0
# 				out=self.U[target_label[:,target_class].long(),target_class,:]
				
# 				##mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
# 				##out=out*mul_factor[:,None]
# 				#out=self.U[target_label[:,target_class].long(),target_class,:]
# 				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
# 				#total=out*mul_factor[:,None]
				
# 				# total=out
# 				# #print(total.shape)
# 				# for ts in targetset[1:]:
# 				# 	ts=torch.Tensor([ts]).long()
# 				# 	out=self.U[target_label[:,ts].long(),ts,:]
# 				# 	#mul_factor=self.uap_weights[target_label[:,ts].long(),ts]
# 				# 	#print(out.shape,mul_factor.shape,(out*mul_factor[:,None]).shape)
# 				# 	#total+=out*mul_factor[:,None]
# 				# 	total+=out

# 				# out = total 
# 			#print(out.shape)
# 			#0/0
# 			#out=out.squeeze()
# 			#print(out.shape)
# 			# print(out[:5,:10])
# 			out=torch.sum(out,1).squeeze()
			
			
# 			# with torch.no_grad():
# 			# 	p=torch.clone(out).detach()
# 			# 	mask=torch.zeros_like(p)
# 			# 	mask[torch.logical_or(p>epsilon_norm,p<-1*epsilon_norm)]=1.0
# 			# 	mask = (mask * p)/epsilon_norm + (1 - mask)

# 			# k = out/mask
# 			#with torch.no_grad():

# 			# a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
# 			# k=out*(epsilon_norm/(a+1e-10))[:,None]
# 			# print('out min max:',torch.min(k),torch.max(k))


# 			# a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
# 			# #t=torch.minimum(torch.Tensor([epsilon_norm]).cuda(),a)
# 			# k=out*(epsilon_norm/(a+1e-30))[:,None]

# 			a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

# 			k=out*(epsilon_norm/(a+1e-10))[:,None]
# 			#k=out 
# 			#print(torch.min(k),torch.max(k))
			

# 			#k=out

# 			#a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
# 			#anew=torch.linalg.vector_norm(k,ord=p_norm,dim=1)
# 			#print('Epsilon norm:',epsilon_norm)
# 			#print('Vector norm:',a[0],anew[0],'Min Max:',torch.min(k),torch.max(k))


# 			#input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
# 			input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)

# 		else:
# 			input_images = x 

# 		#a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

# 		#k=out*(epsilon_norm/(a+1e-10))[:,None]

# 		#print(torch.min(k),torch.max(k))
# 		#k=out


		
		
# 		# 

# 		####
# 		#k=torch.sign(k)*torch.minimum(torch.abs(k),torch.Tensor([epsilon_norm]).cuda())
# 		#k=torch.sum(k,1)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=2))

# 		#0/0

# 		#print(k.shape)
# 		#0/0

# 		#return self.model(x)
# 		# x: input image
# 		# gt: ground truth label
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
# 		#print('Self u shape:',self.U.shape)
# 		# print('\n',torch.sum(self.U[:,:5,:],dim=2))
# 		#print('\n',torch.sum(self.U,dim=2))
# 		#print('\n',self.U[:,:5,:5])
# 		#k=self.U[target_label[:,target_class].long(),(torch.ones_like(target_label[:,target_class])*target_class).long(),:]
		
# 		#print(k[:5,:5])
# 		#print(k.shape)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=1,keepdim=True))
# 		# print(out.shape)
# 		# 0/0
		
# 		# out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 		# out=torch.sum(out,1)
# 		# a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
# 		# # # #k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
# 		# k=out*(epsilon_norm/(a+1e-10))[:,None]
# 		# # print(torch.min(k),torch.max(k))
# 		# # 0/0

		
		
# 		# k=k.view(k.shape[0],x.shape[1],x.shape[2],-1)
# 		# print(k[:5,0,0,:5])
		
		
# 		#input_images = x
		
# 		input_images=torch.clamp(input_images,min=0.0,max=1.0)
# 		#print('\n',torch.min(k).item(),torch.max(k).item(),torch.min(input_images).item(),torch.max(input_images).item())
# 		#0/0
# 		#out,features=self.model(input_images)
# 		#return input_images.detach(),out,features
# 		return self.model(input_images)

# 		# print(torch.linalg.vector_norm(k,ord=2,dim=1))

class Discriminator(torch.nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.disc=nn.Sequential(*[nn.Linear(in_features=2048,out_features=1024,bias=True),
			nn.LeakyReLU(),
			nn.BatchNorm1d(1024),
			nn.Linear(in_features=1024,out_features=512,bias=True),
			nn.LeakyReLU(),
			nn.BatchNorm1d(512),
			nn.Linear(in_features=512,out_features=100,bias=True),
			nn.LeakyReLU(),
			nn.BatchNorm1d(100),
			nn.Linear(in_features=100,out_features=1,bias=True)])

	def forward(self,x):
		out=self.disc(x)
		return out 
		#out = out.mean(0)
		#return out.view(1)
		#return self.disc(x)

# class UAPModel(torch.nn.Module):
# 	def __init__(self,model_params):
# 		super(UAPModel,self).__init__()

# 		self.model=ResNetModel(model_params)
# 		self.model.eval()
# 		checkpoint=torch.load('/mnt/raptor/hassan/AT/weights/voc/common/br_clean/model-696.pt')
# 		#checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/br_l2_1/model-13.pt')
# 		self.model.load_state_dict(checkpoint['model_state'])
		
# 		#self.model.load_state_dict(checkpoint['model_state'])

		
# 		self.num_classes=model_params['num_classes']
		
# 		# freeze this model

# 		for p in self.model.parameters():
# 			p.requires_grad=False

# 		# first row of U would represent all class UAPs with target label 0 and 
# 		# second row would be for all class UAPs of target label 1
# 		#self.U = torch.nn.Parameter(60*torch.rand((2,self.num_classes,3*224*224)).cuda())

# 		d1=torch.randn(1,3*224*224).uniform_(-0.05,0.05)
# 		d1=torch.qr(d1.T)[0].T.cuda()

# 		d2=torch.randn(1,3*224*224).uniform_(-0.05,0.05)
# 		d2=torch.qr(d2.T)[0].T.cuda()


# 		d=torch.stack([d1,d2],dim=0)

# 		print(d.shape)
# 		#z=torch.zeros((self.num_classes,3*224*224),dtype=torch.float32).cuda()
# 		#z[0,:,:]=d
# 		self.U = torch.nn.Parameter(torch.zeros((2,self.num_classes,2,3*224*224),dtype=torch.float32).cuda())
# 		#self.U = torch.nn.Parameter(torch.zeros((2,2,3*224*224),dtype=torch.float32).cuda())
# 		#self.U = torch.nn.Parameter(d)
# 		self.U.requires_grad=True
# 		#self.x=torch.nn.Parameter(torch.randn(1))
# 		#self.x.requires_grad=True
# 		#self.U.register_hook(lambda grad: torch.sign(grad) * 0.1)
		
# 		#self.uap_weights = torch.nn.Parameter(torch.ones((2,self.num_classes),dtype=torch.float32).cuda())
# 		#self.uap_weights = torch.nn.Parameter(torch.rand((2,self.num_classes)).cuda())
# 		#self.uap_weights.requires_grad=True
# 		#self.Normalize_UAP(p_norm=2)

# 		#self.U=torch.nn.utils.weight_norm(U,name='weight',dim=2)
# 		#U_Module = torch.utils.weight_norm(self.U, name='weight',dim=2)
	
# 	def get_norms(self):
# 		#return torch.norm(self.U, dim=2, keepdim=True)
# 		#return self.U
# 		#return torch.sum(self.U)
# 		return torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)
# 		#return torch.linalg.vector_norm(self.U,ord=2,dim=2,keepdim=True)
	
# 	def dotemp(self):
# 		#self.U = torch.nn.Parameter(torch.zeros((2,self.num_classes,3*224*224),dtype=torch.float32).cuda())
# 		#self.U.requires_grad=True
# 		#self.U.register_hook(lambda grad: torch.sign(grad) * 0.1)
		
# 		#self.uap_weights = torch.nn.Parameter(torch.ones((2,self.num_classes),dtype=torch.float32).cuda())
# 		#self.uap_weights.requires_grad=True
# 		return 

# 	#def get_UAPs(self):
# 	def get_params(self):

# 		return self.U
# 		#return self.U


# 	def get_UAPs_features(self):

# 		#k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
# 		#Uaps=U
# 		#k=rescale_to_image_range(self.U,dimval=2)
# 		#k=rescale_to_image_range(self.U*self.uap_weights[:,:,None],dimval=2)
# 		k=rescale_to_image_range(self.U,dimval=2)

# 		# with torch.no_grad():
# 		# 	normval=torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10
# 		# 	#k=self.U/normval
# 		# # k=torch.clamp(k,min=0.0,max=1.0)
# 		# k=self.U/normval
# 		# print('min:',torch.min(k),torch.max(k))
		

# 		all_features=[]
# 		all_outputs=[]
# 		for j in range(1):
# 			tempk=k[j,:,:]
# 			outputs,features=self.model(tempk.view(tempk.shape[0],3,224,224))
# 			all_outputs.append(outputs)
# 			all_features.append(features)
# 			# tempk=k[j,:,:]+1
# 			# k_norm=torch.linalg.vector_norm(tempk,ord=float('inf'),dim=1)
# 			# tempk=torch.clamp(tempk/(k_norm+1e-10)[:,None],min=0.0,max=1.0)
# 			# _,features=self.model(tempk.view(tempk.shape[0],3,224,224))
		
		
# 		return all_outputs,all_features
# 		#return all_features[0],all_features[1]
# 		#return features.view(2,self.num_classes,-1)

# 	def Normalize_UAP(self,p_norm,eps_norm):
# 		#print(self.U.grad[:,:5,:5])
		
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
# 		#print(self.U[:,0,:6])
		
# 		with torch.no_grad():
# 			#self.U.div_(torch.norm(self.U, dim=float('inf'), keepdim=True))
# 			#self.U.div_(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10)

# 			if(p_norm==float('inf')):
# 				#print('normalizing at:',eps_norm)
# 				#0/0
# 				data=self.U.data
# 				norm1=torch.linalg.vector_norm(data[0,:,:,:],ord=float('inf'),dim=2)
# 				norm2=torch.linalg.vector_norm(data[1,:,:,:],ord=float('inf'),dim=2)
				
# 				print(norm1.shape,norm2.shape)

# 				print(data[0,:,:,:].shape)

# 				d1=torch.qr(data[0,:,:,:].T)[0].T.cuda()
# 				d2=torch.qr(data[1,:,:,:].T)[0].T.cuda()

# 				print(d1.shape)
# 				0/0

# 				#print(d1.shape,torch.linalg.vector_norm(data[0,:,:],ord=float('inf'),dim=1).shape)
				
# 				d1=(d1/torch.linalg.vector_norm(d1,ord=float('inf'),dim=1)[:,None])*norm1[:,None]
# 				d2=(d2/torch.linalg.vector_norm(d2,ord=float('inf'),dim=1)[:,None])*norm2[:,None]

# 				#print(d1.shape,d2.shape)
				
# 				# a=torch.linalg.vector_norm(data[0,:,:],ord=p_norm,dim=1)
# 				# d1=data[0,:,:]*(eps_norm/(a+1e-10))[:,None]

# 				# a=torch.linalg.vector_norm(data[1,:,:],ord=p_norm,dim=1)
# 				# d2=data[1,:,:]*(eps_norm/(a+1e-10))[:,None]

# 				d1=normalize_vec(d1,max_norm=eps_norm,norm_p=p_norm)				
# 				d2=normalize_vec(d2,max_norm=eps_norm,norm_p=p_norm)				

# 				self.U.data[0,:,:]=d1
# 				self.U.data[1,:,:]=d2

# 				self.x.data=self.x.data/(torch.sum(self.x.data)+1e-9)


# 				#self.U.data=normalize_vec(self.U,max_norm=eps_norm,norm_p=p_norm)
				

			
# 			else:
# 				a=torch.linalg.vector_norm(self.U,ord=p_norm,dim=2)
# 				#t=torch.minimum(torch.Tensor([eps_norm]).cuda(),a)
# 				#self.U.data=self.U*(t/(a+1e-30))[:,:,None]
# 				self.U.data=self.U*(eps_norm/(a+1e-10))[:,:,None]

			
# 			#self.U.data=self.U*(torch.min(eps_norm,a+1e-10)/(a+1e-10))[:,:,None]
# 			#print(k.shape)
# 			#print(torch.linalg.vector_norm(k,ord=p_norm,dim=2))

# 			#0/0
# 			#self.U.data=normalize_vec(self.U.view(-1,3,224,224),max_norm=1.0,norm_p=float(2))
# 			#self.uap_weights.data=self.uap_weights/torch.sum(self.uap_weights,dim=1)[:,None]
			

# 		#self.U = F.normalize(self.U, p=2.0, dim=2)
# 		#torch.nn.utils.weight_norm(self.U,name='weight',dim=2)
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		

# 	def forward(self,x,par,targetset=[]):
# 		epsilon_norm=par['epsilon_norm']
# 		#return self.model(x)
# 		#k=epsilon_norm*self.U[target_label[:,target_class].long(),target_class,:]
# 		# k=self.U[target_label[:,target_class].long(),target_class,:]
		

# 		# print(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2))
# 		# 0/0
		
# 		#print(out.shape)
# 		if(epsilon_norm>0.0):
# 			p_norm,target_label,target_class,combine_uaps=par['p_norm'],par['target_label'],par['target_class'],par['combine_uaps']
# 			if(combine_uaps):
# 				out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 				#print(torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],1).shape)
# 				##mul_factor=self.uap_weights[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
# 				##out=out*mul_factor[:,:,None]
# 				0/0
# 			else:
# 				#tempU=torch.sum(self.U*torch.randn(20)[:,None].cuda(),dim=0)
# 				#out=(self.U[target_label[:,target_class].long(),:,:].squeeze(dim=1))
# 				#print(out.shape)
# 				randvals=(-2*torch.randn(2)+1)[None,:,None].cuda()
# 				out=(self.U[target_label[:,target_class].long(),:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)
# 				#print(out.shape)
# 				#0/0
# 				#out=(self.U[target_label[:,target_class].long(),:,:].squeeze(dim=1)*self.x[:,None]).sum(dim=1)#.view(x.shape[0],3,224,224)

# 				#print(x.shape,out.shape)
# 				#print(out.view(3,224,224).expand(x.shape[0],-1).shape)
# 				#out = tempU.view(3,224,224).expand(x.shape[0],-1,-1,-1)
				
# 				#0/0

# 				# 0/0
# 				# out=self.U[target_label[:,target_class].long(),target_class,:]
				
# 				##mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
# 				##out=out*mul_factor[:,None]
# 				#out=self.U[target_label[:,target_class].long(),target_class,:]
# 				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
# 				#total=out*mul_factor[:,None]
				
# 				# total=out
# 				# #print(total.shape)
# 				# for ts in targetset[1:]:
# 				# 	ts=torch.Tensor([ts]).long()
# 				# 	out=self.U[target_label[:,ts].long(),ts,:]
# 				# 	#mul_factor=self.uap_weights[target_label[:,ts].long(),ts]
# 				# 	#print(out.shape,mul_factor.shape,(out*mul_factor[:,None]).shape)
# 				# 	#total+=out*mul_factor[:,None]
# 				# 	total+=out

# 				# out = total 
# 			#print(out.shape)
# 			#0/0
# 			#out=out.squeeze()
# 			#print(out.shape)
# 			# print(out[:5,:10])

# 			# ####################
# 			#out=torch.sum(out,1).squeeze()
			
# 			#print(out.shape)
# 			a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
# 			#print(a.shape)
# 			#t=torch.minimum(torch.Tensor([epsilon_norm]).cuda(),a)
# 			k=out*(epsilon_norm/(a+1e-10))[:,None]
# 			# ########################

# 			# with torch.no_grad():
# 			# 	p=torch.clone(out).detach()
# 			# 	mask=torch.zeros_like(p)
# 			# 	mask[torch.logical_or(p>epsilon_norm,p<-1*epsilon_norm)]=1.0
# 			# 	mask = (mask * p)/epsilon_norm + (1 - mask)

# 			# k = out/mask
# 			#with torch.no_grad():
			
# 			##a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
# 			##k=out*(epsilon_norm/(a+1e-10))[:,None]
# 			#k=out

# 			#a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
# 			#anew=torch.linalg.vector_norm(k,ord=p_norm,dim=1)
# 			#print('Epsilon norm:',epsilon_norm)
# 			#print('Vector norm:',a[0],anew[0],'Min Max:',torch.min(k),torch.max(k))


# 			#input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
			
# 			#k=out
# 			#print(k.shape,x.shape)
# 			input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
# 			#input_images = x + k

# 		else:
# 			input_images = x 

# 		#a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

# 		#k=out*(epsilon_norm/(a+1e-10))[:,None]

# 		#print(torch.min(k),torch.max(k))
# 		#k=out


		
		
# 		# 

# 		####
# 		#k=torch.sign(k)*torch.minimum(torch.abs(k),torch.Tensor([epsilon_norm]).cuda())
# 		#k=torch.sum(k,1)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=2))

# 		#0/0

# 		#print(k.shape)
# 		#0/0

# 		#return self.model(x)
# 		# x: input image
# 		# gt: ground truth label
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
# 		#print('Self u shape:',self.U.shape)
# 		# print('\n',torch.sum(self.U[:,:5,:],dim=2))
# 		#print('\n',torch.sum(self.U,dim=2))
# 		#print('\n',self.U[:,:5,:5])
# 		#k=self.U[target_label[:,target_class].long(),(torch.ones_like(target_label[:,target_class])*target_class).long(),:]
		
# 		#print(k[:5,:5])
# 		#print(k.shape)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=1,keepdim=True))
# 		# print(out.shape)
# 		# 0/0
		
# 		# out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 		# out=torch.sum(out,1)
# 		# a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
# 		# # # #k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
# 		# k=out*(epsilon_norm/(a+1e-10))[:,None]
# 		# # print(torch.min(k),torch.max(k))
# 		# # 0/0

		
		
# 		# k=k.view(k.shape[0],x.shape[1],x.shape[2],-1)
# 		# print(k[:5,0,0,:5])
		
		
# 		#input_images = x
		
# 		input_images=torch.clamp(input_images,min=0.0,max=1.0)
# 		#print('\n',torch.min(k).item(),torch.max(k).item(),torch.min(input_images).item(),torch.max(input_images).item())
# 		#0/0
# 		#out,features=self.model(input_images)
# 		#return input_images.detach(),out,features
# 		return self.model(input_images)

# 		# print(torch.linalg.vector_norm(k,ord=2,dim=1))


class UAPModel(torch.nn.Module):
	def __init__(self,model_params):
		super(UAPModel,self).__init__()

		self.model=ResNetModel(model_params)
		self.model.eval()
		checkpoint=torch.load('/mnt/raptor/hassan/AT/weights/voc/common/br_clean/model-696.pt')
		#checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/br_l2_1/model-13.pt')
		self.model.load_state_dict(checkpoint['model_state'])
		
		#self.model.load_state_dict(checkpoint['model_state'])

		
		self.num_classes=model_params['num_classes']
		
		# freeze this model

		for p in self.model.parameters():
			p.requires_grad=False

		# first row of U would represent all class UAPs with target label 0 and 
		# second row would be for all class UAPs of target label 1

		self.U = torch.nn.Parameter(torch.zeros((2,self.num_classes,2,3*224*224),dtype=torch.float32).cuda())
		#self.U = torch.nn.Parameter(torch.zeros((2,2,3*224*224),dtype=torch.float32).cuda())
		#self.U = torch.nn.Parameter(d)
		self.U.requires_grad=True
	
	def get_norms(self):
		#return torch.norm(self.U, dim=2, keepdim=True)
		#return self.U
		#return torch.sum(self.U)
		return torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)
		#return torch.linalg.vector_norm(self.U,ord=2,dim=2,keepdim=True)
	
	
	def get_params(self):

		return self.U
		#return self.U


	def get_UAPs_features(self):

		#k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
		#Uaps=U
		#k=rescale_to_image_range(self.U,dimval=2)
		#k=rescale_to_image_range(self.U*self.uap_weights[:,:,None],dimval=2)
		k=rescale_to_image_range(self.U,dimval=2)

		# with torch.no_grad():
		# 	normval=torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10
		# 	#k=self.U/normval
		# # k=torch.clamp(k,min=0.0,max=1.0)
		# k=self.U/normval
		# print('min:',torch.min(k),torch.max(k))
		

		all_features=[]
		all_outputs=[]
		for j in range(1):
			tempk=k[j,:,:]
			outputs,features=self.model(tempk.view(tempk.shape[0],3,224,224))
			all_outputs.append(outputs)
			all_features.append(features)
			# tempk=k[j,:,:]+1
			# k_norm=torch.linalg.vector_norm(tempk,ord=float('inf'),dim=1)
			# tempk=torch.clamp(tempk/(k_norm+1e-10)[:,None],min=0.0,max=1.0)
			# _,features=self.model(tempk.view(tempk.shape[0],3,224,224))
		
		
		return all_outputs,all_features
		#return all_features[0],all_features[1]
		#return features.view(2,self.num_classes,-1)

	def Normalize_UAP(self,p_norm,eps_norm):
		#print(self.U.grad[:,:5,:5])
		
		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		#print(self.U[:,0,:6])
		
		with torch.no_grad():
			#self.U.div_(torch.norm(self.U, dim=float('inf'), keepdim=True))
			#self.U.div_(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10)

			if(p_norm==float('inf')):
				#print('normalizing at:',eps_norm)
				#0/0
				data=self.U.data
				norm1=torch.linalg.vector_norm(data[0,:,:,:],ord=float('inf'),dim=2)
				norm2=torch.linalg.vector_norm(data[1,:,:,:],ord=float('inf'),dim=2)
				
				#print(norm1.shape,norm2.shape)
				#print(data[0,:,:,:].shape)

				d1=torch.transpose(torch.qr(torch.transpose(data[0,:,:,:],1,2))[0],2,1).cuda()
				d2=torch.transpose(torch.qr(torch.transpose(data[1,:,:,:],1,2))[0],2,1).cuda()


				#print(d1.shape,torch.linalg.vector_norm(data[0,:,:],ord=float('inf'),dim=1).shape)
				#print(d1.shape,d2.shape)
				#print(torch.linalg.vector_norm(d1,ord=float('inf'),dim=2).shape)
				#print()

				d1=(d1/torch.linalg.vector_norm(d1,ord=float('inf'),dim=2)[:,:,None])*norm1[:,:,None]
				d2=(d2/torch.linalg.vector_norm(d2,ord=float('inf'),dim=2)[:,:,None])*norm2[:,:,None]
				
				#print(d1.shape,d2.shape)
				
				# a=torch.linalg.vector_norm(data[0,:,:],ord=p_norm,dim=1)
				# d1=data[0,:,:]*(eps_norm/(a+1e-10))[:,None]

				# a=torch.linalg.vector_norm(data[1,:,:],ord=p_norm,dim=1)
				# d2=data[1,:,:]*(eps_norm/(a+1e-10))[:,None]

				d1=normalize_vec(d1,max_norm=eps_norm,norm_p=p_norm)				
				d2=normalize_vec(d2,max_norm=eps_norm,norm_p=p_norm)				


				self.U.data[0,:,:,:]=d1
				self.U.data[1,:,:,:]=d2

				#self.x.data=self.x.data/(torch.sum(self.x.data)+1e-9)


				#self.U.data=normalize_vec(self.U,max_norm=eps_norm,norm_p=p_norm)
				

			
			else:
				a=torch.linalg.vector_norm(self.U,ord=p_norm,dim=2)
				#t=torch.minimum(torch.Tensor([eps_norm]).cuda(),a)
				#self.U.data=self.U*(t/(a+1e-30))[:,:,None]
				self.U.data=self.U*(eps_norm/(a+1e-10))[:,:,None]

			
			l=[]
			for i in range(6):
				for j in range(6):
					l.append(np.rad2deg(subspace_angles(self.U[1,i,:,:].T.detach().cpu().numpy(),self.U[1,j,:,:].T.detach().cpu().numpy())).tolist())

			print(l)

			#self.U.data=self.U*(torch.min(eps_norm,a+1e-10)/(a+1e-10))[:,:,None]
			#print(k.shape)
			#print(torch.linalg.vector_norm(k,ord=p_norm,dim=2))

			#0/0
			#self.U.data=normalize_vec(self.U.view(-1,3,224,224),max_norm=1.0,norm_p=float(2))
			#self.uap_weights.data=self.uap_weights/torch.sum(self.uap_weights,dim=1)[:,None]
			

		#self.U = F.normalize(self.U, p=2.0, dim=2)
		#torch.nn.utils.weight_norm(self.U,name='weight',dim=2)
		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		

	def forward(self,x,par,targetset=[]):
		epsilon_norm=par['epsilon_norm']

		classes=[0,15,16]
		
		Uk=[self.U[0,c,:,:] for c in classes]
		Us=[u/torch.linalg.vector_norm(u,ord=2,dim=1)[:,None] for u in Uk]

		#off = torch.cat([self.U[0,0,:,:],self.U[0,5,:,:]],dim=0).T
		#off = torch.cat([self.U[0,c,:,:] for c in classes],dim=0).T
		off = torch.cat([Us[c] for c in range(len(classes))],dim=0).T
		u,s,v=torch.svd(off)
		off=u[:,0].cuda()

		Uk=[self.U[1,c,:,:] for c in classes]
		Us=[u/torch.linalg.vector_norm(u,ord=2,dim=1)[:,None] for u in Uk]
		on = torch.cat([Us[c] for c in range(len(classes))],dim=0).T
		u,s,v=torch.svd(on)
		on=u[:,0].cuda()

		# np.save('/mnt/raptor/hassan/data/model_uniform_subspace_close.npy',self.U.data.detach().cpu().numpy())
		# 0/0
		#return self.model(x)
		#k=epsilon_norm*self.U[target_label[:,target_class].long(),target_class,:]
		# k=self.U[target_label[:,target_class].long(),target_class,:]
		

		# from scipy.linalg import subspace_angles
		# #print(class0.shape,class5.shape)
		# #print(self.U[1,0,:,:].squeeze().detach().T.shape)
		# for j in range(10):
		# 	for i in range(10):
		# 		class0=self.U[1,j,:,:].squeeze()
		# 		class1=self.U[1,i,:,:].squeeze()
		# 		#print(class0)
		# 		#print(class1)
		# 		class0=torch.qr(class0.T)[0].T
		# 		class1=torch.qr(class1.T)[0].T
		# 		#class0=.squeeze()
		# 		#class1=self.U[1,i,:,:].squeeze()

		# 		# class0=class0/torch.linalg.vector_norm(class0,ord=2,dim=1)[:,None]
		# 		# class1=class1/torch.linalg.vector_norm(class1,ord=2,dim=1)[:,None]
		# 		class0=class0.T 
		# 		class1=class1.T 
		# 		#print(class0.shape,class1.shape)

		# 		print(j,i,np.rad2deg(subspace_angles(class0.detach().cpu().numpy(),class1.detach().cpu().numpy())))

		# 0/0
		
		#print(out.shape)
		if(epsilon_norm>0.0):
			p_norm,target_label,target_class,combine_uaps=par['p_norm'],par['target_label'],par['target_class'],par['combine_uaps']
			if(combine_uaps):
				out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
				#print(torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],1).shape)
				##mul_factor=self.uap_weights[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
				##out=out*mul_factor[:,:,None]
				0/0
			else:

				#currenttargets=[0,8]
				out=torch.zeros_like(x).view(x.shape[0],-1)
				#randvals=(-2*torch.rand(2)+1)[None,:,None].cuda()
				#randvals=torch.tensor([-0.762,-0.648])[None,:,None].cuda()
				#for 12 randvals=torch.tensor([-0.939,0.345])[None,:,None].cuda()
				#randvals=torch.tensor([0.336,-0.942])[None,:,None].cuda()
				#randvals=torch.tensor([-0.3128,0.368])[None,:,None].cuda()

				#randvals=torch.tensor([-0.9725,0.233])[None,:,None].cuda()
				# randvals=torch.tensor([0.9252,-0.3795])[None,:,None].cuda()
				#randvals=torch.tensor([0.999,0.047])[None,:,None].cuda()
				#randvals=torch.tensor([-0.99,-1.395])[None,:,None].cuda()
				#randvals=torch.tensor([-0.949,0.3161])[None,:,None].cuda()
				#randvals=torch.tensor([-0.987,-0.1599])[None,:,None].cuda()
				#randvals=torch.tensor([0.459,-0.8882])[None,:,None].cuda()

				# randvals=torch.tensor([0.4924,-0.1824])[None,:,None].cuda()
				#randvals=torch.tensor([-0.7902,0.1878])[None,:,None].cuda()
				# randvals=torch.tensor([-0.5175,-0.4104])[None,:,None].cuda() #this worked really well for 0,14,16
				# randvals=torch.tensor([0.7169,0.03244])[None,:,None].cuda()
				# out=out+(self.U[target_label[:,0].long(),0,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				# #randvals=torch.tensor([0.7957,-0.207])[None,:,None].cuda()
				# #randvals=torch.tensor([0.5034,-0.1220])[None,:,None].cuda()
				# randvals=torch.tensor([0.6484,0.2678])[None,:,None].cuda()
				# out=out+(self.U[target_label[:,14].long(),14,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				#randvals=torch.tensor([-0.0638,0.21])[None,:,None].cuda() #this worked for 0,14,16
				#randvals=torch.tensor([0.6484, -0.0471])[None,:,None].cuda() #this worked really well for 0,14,16
				#randvals=torch.tensor([0.7169,0.3244])[None,:,None].cuda()
				#randvals=torch.tensor([-0.18824,0.01085])[None,:,None].cuda()
				
				# randvals=torch.tensor([0.717,0.032])[None,:,None].cuda()
				# # #randvals=torch.tensor([-0.635,-0.278])[None,:,None].cuda()
				# # #randvals=torch.tensor([-0.166,0.009])[None,:,None].cuda()
				#randvals=torch.tensor([0.1891,-0.0144])[None,:,None].cuda()
				#randvals=torch.tensor([0.041,-0.278])[None,:,None].cuda()
				# randvals=torch.tensor([0.295,-0.643])[None,:,None].cuda()
				#randvals=torch.tensor([-0.2945,0.643])[None,:,None].cuda()
				# randvals=torch.tensor([0.2945,-0.2379])[None,:,None].cuda()
				# out=out+(self.U[target_label[:,0].long(),0,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				# randvals=torch.tensor([0.670,0.029])[None,:,None].cuda()
				# # #randvals=torch.tensor([0.703,-0.154])[None,:,None].cuda()
				# # #randvals=torch.tensor([-0.134,0.049])[None,:,None].cuda()
				#randvals=torch.tensor([-0.2945,0.643])[None,:,None].cuda()
				#randvals=torch.tensor([0.295,-0.643])[None,:,None].cuda()
				#randvals=torch.tensor([0.2945,-0.2379])[None,:,None].cuda()
				#randvals=(-2*torch.rand(2)+1)[None,:,None].cuda()
				# randvals=torch.tensor([-0.1,0.698])[None,:,None].cuda()
				# out=out+(self.U[target_label[:,15].long(),15,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				#randvals=torch.tensor([0.4975,0.00376])[None,:,None].cuda()
				# randvals=torch.tensor([-0.62,-0.33])[None,:,None].cuda()
				# #randvals=torch.tensor([0.273,0.962])[None,:,None].cuda()
				# k=(self.U[target_label[:,0].long(),0,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)
				# a=torch.linalg.vector_norm(k,ord=float('inf'),dim=1)
				# out=out+k*(0.1/(a+1e-10))[:,None]

				#randvals=torch.tensor([0.51,0.0072])[None,:,None].cuda()
				# randvals=torch.tensor([-0.62,-0.33])[None,:,None].cuda()
				# #randvals=torch.tensor([0.273,0.962])[None,:,None].cuda()
				# k=(self.U[target_label[:,5].long(),5,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)
				# a=torch.linalg.vector_norm(k,ord=float('inf'),dim=1)
				# out=out+k*(0.1/(a+1e-10))[:,None]
				out=torch.zeros((x.shape[0],3*224*224),dtype=torch.float32).cuda()
				out[target_label[:,0]==0.0]=off
				out[target_label[:,0]==1.0]=on  


				#randvals=torch.tensor([-0.1,0.698])[None,:,None].cuda()
				# randvals=torch.tensor([0.6838,-0.18])[None,:,None].cuda()
				#randvals=torch.tensor([-0.7045,0.061])[None,:,None].cuda()
				#randvals=torch.tensor([0.51,-0.076])[None,:,None].cuda()
				#randvals=torch.tensor([-0.505,-0.0103])[None,:,None].cuda()
				# randvals=torch.tensor([0.341,0.0573])[None,:,None].cuda()
				# #randvals=torch.tensor([0.273,0.962])[None,:,None].cuda()
				# k=(self.U[target_label[:,15].long(),15,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)
				# a=torch.linalg.vector_norm(k,ord=float('inf'),dim=1)
				# out=out+k*(0.1/(a+1e-10))[:,None]
				# a=torch.linalg.vector_norm(k,ord=float('inf'),dim=1)
				# #print(a.shape)
				# #t=torch.minimum(torch.Tensor([epsilon_norm]).cuda(),a)
				# out=out+k*(0.1/(a+1e-10))[:,None]

				#randvals=torch.tensor([-0.188,0.118])[None,:,None].cuda()
				#randvals=torch.tensor([0.011,0.012])[None,:,None].cuda()
				#randvals=torch.tensor([-0.188,0.011])[None,:,None].cuda()
				#randvals=torch.tensor([0.717,0.032])[None,:,None].cuda()
				#randvals=torch.tensor([-0.037,0.959])[None,:,None].cuda()
				#randvals=torch.tensor([-0.839,-0.366])[None,:,None].cuda()
				#randvals=torch.tensor([0.041,-0.278])[None,:,None].cuda()
				# randvals=torch.tensor([0.1891,-0.0144])[None,:,None].cuda()
				#randvals=torch.tensor([-0.1,0.698])[None,:,None].cuda()
				#randvals=torch.tensor([0.6838,-0.18])[None,:,None].cuda()
				#randvals=torch.tensor([-0.7045,0.061])[None,:,None].cuda()
				#randvals=torch.tensor([0.56,0.053])[None,:,None].cuda()
				# randvals=torch.tensor([-0.4957,-0.00229])[None,:,None].cuda()
				# randvals=torch.tensor([0.66,0.17])[None,:,None].cuda()
				# #randvals=torch.tensor([0.273,0.962])[None,:,None].cuda()
				# k=(self.U[target_label[:,16].long(),16,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)
				# a=torch.linalg.vector_norm(k,ord=float('inf'),dim=1)
				# out=out+k*(0.1/(a+1e-10))[:,None]
				# a=torch.linalg.vector_norm(k,ord=float('inf'),dim=1)
				# out=out+k*(0.1/(a+1e-10))[:,None]

				# randvals=torch.tensor([0.5792,0.0071])[None,:,None].cuda()
				# #randvals=torch.tensor([0.273,0.962])[None,:,None].cuda()
				# out=out+(self.U[target_label[:,17].long(),17,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)
				# a=torch.linalg.vector_norm(k,ord=float('inf'),dim=1)
				# out=out+k*(0.1/(a+1e-10))[:,None]
				#out=out+(self.U[target_label[:,18].long(),18,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				#randvals=torch.tensor([-0.99,-0.139])[None,:,None].cuda()
				#out=out+(self.U[target_label[:,11].long(),11,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				# for ks in currenttargets:
				# 	randvals=(-2*torch.rand(2)+1)[None,:,None].cuda()
				# 	print(randvals)
				# 	out=out+(self.U[target_label[:,ks].long(),ks,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				#out=(self.U[target_label[:,0].long(),0,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				#randvals=(-2*torch.rand(2)+1)[None,:,None].cuda()
				#out=out+(self.U[target_label[:,14].long(),14,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				#randvals=(-2*torch.rand(2)+1)[None,:,None].cuda()
				#out=out+(self.U[target_label[:,16].long(),16,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)

				#class0=torch.sum(self.U[1,0,:,:],dim=0)
				#class1=torch.sum(self.U[1,5,:,:],dim=0)
				#sumval=class0+class1


				#tempU=torch.sum(self.U*torch.randn(20)[:,None].cuda(),dim=0)
				#out=(self.U[target_label[:,target_class].long(),:,:].squeeze(dim=1))
				#print(out.shape)
				
				##################
				#randvals=(-2*torch.rand(2)+1)[None,:,None].cuda()
				#out=(self.U[target_label[:,target_class].long(),target_class,:,:].squeeze(dim=1)*randvals).sum(dim=1)#.view(x.shape[0],3,224,224)
				##################

				#print(sumval.shape)
				#out = sumval.expand(target_label.shape[0],-1)
				#print(out.shape)
				#0/0
				#print(out.shape)
				#0/0
				#out=(self.U[target_label[:,target_class].long(),:,:].squeeze(dim=1)*self.x[:,None]).sum(dim=1)#.view(x.shape[0],3,224,224)

				#print(x.shape,out.shape)
				#print(out.view(3,224,224).expand(x.shape[0],-1).shape)
				#out = tempU.view(3,224,224).expand(x.shape[0],-1,-1,-1)
				
				#0/0

				# 0/0
				# out=self.U[target_label[:,target_class].long(),target_class,:]
				
				##mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
				##out=out*mul_factor[:,None]
				#out=self.U[target_label[:,target_class].long(),target_class,:]
				#mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
				#total=out*mul_factor[:,None]
				
				# total=out
				# #print(total.shape)
				# for ts in targetset[1:]:
				# 	ts=torch.Tensor([ts]).long()
				# 	out=self.U[target_label[:,ts].long(),ts,:]
				# 	#mul_factor=self.uap_weights[target_label[:,ts].long(),ts]
				# 	#print(out.shape,mul_factor.shape,(out*mul_factor[:,None]).shape)
				# 	#total+=out*mul_factor[:,None]
				# 	total+=out

				# out = total 
			#print(out.shape)
			#0/0
			#out=out.squeeze()
			#print(out.shape)
			# print(out[:5,:10])

			# ####################
			#out=torch.sum(out,1).squeeze()
			
			#print(out.shape)
			a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
			#print(a.shape)
			#t=torch.minimum(torch.Tensor([epsilon_norm]).cuda(),a)
			k=out*(epsilon_norm/(a+1e-10))[:,None]
			# ########################

			# with torch.no_grad():
			# 	p=torch.clone(out).detach()
			# 	mask=torch.zeros_like(p)
			# 	mask[torch.logical_or(p>epsilon_norm,p<-1*epsilon_norm)]=1.0
			# 	mask = (mask * p)/epsilon_norm + (1 - mask)

			# k = out/mask
			#with torch.no_grad():
			
			##a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
			##k=out*(epsilon_norm/(a+1e-10))[:,None]
			#k=out

			#a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
			#anew=torch.linalg.vector_norm(k,ord=p_norm,dim=1)
			#print('Epsilon norm:',epsilon_norm)
			#print('Vector norm:',a[0],anew[0],'Min Max:',torch.min(k),torch.max(k))


			#input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
			
			#k=out
			#print(k.shape,x.shape)
			input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
			#input_images = x + k

		else:
			input_images = x 

		#a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

		#k=out*(epsilon_norm/(a+1e-10))[:,None]

		#print(torch.min(k),torch.max(k))
		#k=out


		
		
		# 

		####
		#k=torch.sign(k)*torch.minimum(torch.abs(k),torch.Tensor([epsilon_norm]).cuda())
		#k=torch.sum(k,1)
		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=2))

		#0/0

		#print(k.shape)
		#0/0

		#return self.model(x)
		# x: input image
		# gt: ground truth label
		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
		#print('Self u shape:',self.U.shape)
		# print('\n',torch.sum(self.U[:,:5,:],dim=2))
		#print('\n',torch.sum(self.U,dim=2))
		#print('\n',self.U[:,:5,:5])
		#k=self.U[target_label[:,target_class].long(),(torch.ones_like(target_label[:,target_class])*target_class).long(),:]
		
		#print(k[:5,:5])
		#print(k.shape)
		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=1,keepdim=True))
		# print(out.shape)
		# 0/0
		
		# out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
		# out=torch.sum(out,1)
		# a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
		# # # #k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
		# k=out*(epsilon_norm/(a+1e-10))[:,None]
		# # print(torch.min(k),torch.max(k))
		# # 0/0

		
		
		# k=k.view(k.shape[0],x.shape[1],x.shape[2],-1)
		# print(k[:5,0,0,:5])
		
		
		#input_images = x
		
		input_images=torch.clamp(input_images,min=0.0,max=1.0)
		#print('\n',torch.min(k).item(),torch.max(k).item(),torch.min(input_images).item(),torch.max(input_images).item())
		#0/0
		#out,features=self.model(input_images)
		#return input_images.detach(),out,features
		return self.model(input_images)

		# print(torch.linalg.vector_norm(k,ord=2,dim=1))


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
	
	def forward(self,x):
		globalfeature=self.prelu1(self.basemodel(x))
		globalfeature=torch.flatten(globalfeature,1)
		#globalfeature.requires_grad=True
		output=self.classifier(globalfeature)
		#print('Global feature:',torch.sum(globalfeature),torch.sum(output))
		return output,globalfeature


