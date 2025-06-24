import os
import sys
sys.path.append('./')
from utils.utility import *
from utils.confparser import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm 
from Models import *
#from Datasets.NUSDataset import NUSImageDataset as ImageDataset
from Datasets import * 
import re
from Logger.Logger import *
import configparser
import ast, json
import argparse
import pickle
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from scipy.linalg import subspace_angles
from utils.modelfactory import *
sys.path.append('ASL/')
#from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from src.models import create_model
pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/voc.ini')
parser.add_argument('--mode')
parser.add_argument('expname')


class Tester:
	def __init__(self,configfile,experiment_name,mode):
		self.mode=mode
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':mode})	
		# self.passedtargetsize=passedtargetsize
		#self.parse_data(configfile,experiment_name,losstype)
	
	def set_bn_eval(self,module):
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()

	def test(self):
		params=self.parseddata.build()
		
		####################################################################################
		# Retrieve all the variables from config file

		device=torch.device('cuda:0')
		self.logger=params['logger']
		writer=params['writer']

		datasetname=params['dataset_name']
		# epsilon_norm=float(ast.literal_eval(params['eps_norm']))
		p_norm=float(params['p_norm'])
		
		epsilon_norm=float(params['eps_norm'])
		model_path=params['main_model_path']
		model_name=params['model_name']
		image_size=float(params['image_size'])
			
		num_classes=float(params['num_classes'])

		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		
		weights_dir=params['weights_dir']
		target_classes_dir=params['target_classes_dir']
		method_name=params['method_name']
		batchsize = params['batch_size']
		targetsize=int(params['targetsize'])

		####################################################################################

		# Create model given by model_name
		model=createmodel(model_name,model_path,self.logger,datasetname ,argparse.Namespace(**params))


		self.logger.write('\nTarget size:',targetsize)
		self.logger.write('\nMethod:',method_name)
		self.logger.write('\nDataset:',datasetname)
		self.logger.write('\nModel name:',model_name)


		# Given C classes of a dataset, we choose a subset C' of classes for experiments. All the combinations of different sizes (1, 2, 3, ...) are chosen from the classes in C'.
		# listalltargetclasses1 are just the classes C'.
		# For example, assume C' = {Person, Car, Cat, Dog, Table}. Then some combinations made from those classes of size 3 are: 
		# targetclasses = {{Person, Car, Cat}, {Person, Car, Dog}, {Person, Car, Table}}

		listalltargetclasses1=sum(pickle.load(open(os.path.join(target_classes_dir,str(1)),'rb'))['target_classes'],[]) 
		# We need universal perturbation for each class in C'
		num_universals=len(listalltargetclasses1)

		# We store C' and their different combinations in a file. targetdata are those combinations.
		targetdata=pickle.load(open(os.path.join(target_classes_dir,str(targetsize)),'rb'))


		if method_name=='oracle':
			print('Loading weights:',params['checkpoint_load_path'])
			U=torch.load(params['checkpoint_load_path'])['model_state']
		elif method_name=='or-c':
			# print('loading coefficients:',params['oracle1_weight'])
			U_coeff=torch.load(params['checkpoint_load_path'])['model_state']
			oracle1 = torch.load(params['oracle1_weight'])['model_state'][0,:,:].unsqueeze(dim=0)

		elif method_name=='or-s':
			# print('Loading weights:',params['oracle1_weight'])
			U = torch.load(params['oracle1_weight'])['model_state'][0,:,:].unsqueeze(dim=0)	
		# elif method_name=='cgan_derived' or method_name=='cgan':
		elif method_name=='nag':
			per_class_dim=100
			gan_vec_size=per_class_dim+num_universals
			ganU=netG(gan_vec_size).to(device)

			loadedweights=torch.load(params['checkpoint_load_path'])['model_state']
			# ganU.load_state_dict(loadedweights[0])
			ganU.load_state_dict(loadedweights)

		elif method_name=='cmlu_b' or method_name=='cmlu_a':
			U = torch.load(params['checkpoint_load_path'])['model_state']

		else:
			raise RuntimeError("Please choose the method name as one of the given.")

		
		model=model.to(device)

		
		targetclasses=targetdata['target_classes']
		
		# To evaluate the attack, we need to know the images that contain the specific classes that we want to attack. We store the image ids and their respective labels in separate files, so we can retrieve them quickly and easily.
		test_img_ids=pickle.load(open(ast.literal_eval(params['test_pred_img_ids_file']),'rb'))
		test_labels=np.load(ast.literal_eval(params['test_pred_file']))
		test_indices_dir=ast.literal_eval(params['test_indices_dir'])

		# To evaluate attacks on different classes or their combinations, we need to have dataloaders for each combination of the classes.

		alltestloaders=[]
		testdatasetlengths=[]
		for tindex,jk in enumerate(tqdm(targetclasses)):
			test_indices=np.load(os.path.join(test_indices_dir,str(targetsize),str('_'.join([str(v) for v in jk]))+'.npy'))
			
			testdatasetparams={
			'input_size':ast.literal_eval(params['image_size']),
			'images_dir':ast.literal_eval(params['test_images_dir']),
			'all_img_ids':np.array(test_img_ids)[test_indices].tolist(),
			'all_labels':test_labels[test_indices,:]
			}
			test_dataset=SelectiveDataset(testdatasetparams)
			testdatasetlengths.append(len(test_dataset))

			test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=int(batchsize), shuffle=False, num_workers=0)
			alltestloaders.append(test_dataloader)
		

		print('Data Loaders:',len(alltestloaders))
		print('Length of test datasets:',testdatasetlengths)

		# print(targetclasses)
		# print(listalltargetclasses1)
		
		allcombineduaps=[]
		
		if method_name=='or-C':
			for tindex,target_class in enumerate(targetclasses):
				
				# U_\Omega = \sum_{i \in \Omega} c_i * u_i where c_i are coefficients that we have learnt through training.
				tempU=(U_coeff[:,:,tindex].unsqueeze(dim=-1)*oracle1[:,[listalltargetclasses1.index(v) for v in target_class],:]).view(-1,oracle1.shape[-1]).sum(dim=0)
				# tempU=torch.matmul(U_coeff[:,tindex],oracle1[0,[listalltargetclasses1.index(t) for t in target_class],:])

				a=torch.linalg.vector_norm(tempU,ord=float('inf'))
			
				t=torch.minimum(torch.Tensor([epsilon_norm]).to(device),a)
			
				tempU=tempU*(t/(a+1e-9))
				allcombineduaps.append(torch.clone(tempU))
		
		elif method_name=='oracle':
			# Exponential complexity method where we evaluate UAPs learnt directly on the target class combination
			for tindex,target_class in enumerate(targetclasses):

				allcombineduaps.append(U[:,tindex,:])

				

		elif method_name=='or-S' or method_name=='cmlu_a' or method_name=='cmlu_b':
			# Naive summation and CMLU
			for tindex,target_class in enumerate(targetclasses):

				tempU=torch.zeros_like(U[0,0,:])
				for t in target_class:
				
					tempU+=U[0,listalltargetclasses1.index(t),:]
					
				
				a=torch.linalg.vector_norm(tempU,ord=float('inf'))
			
				t=torch.minimum(torch.Tensor([epsilon_norm]).to(device),a)
			
				tempU=tempU*(t/(a+1e-9))
				allcombineduaps.append(torch.clone(tempU))

		elif method_name=='nag':
			# Generative model.

			allcombineduaps=[]
			# The noisevec will have the top part as noise and the last part as the code of the target classes.
			noisevec=torch.rand((len(targetclasses),gan_vec_size))*2-1
			labelvec=np.zeros(shape=(len(targetclasses),len(listalltargetclasses1)),dtype=np.float32)
			
			for tindex,target_class in enumerate(targetclasses):
				labelvec[tindex,[listalltargetclasses1.index(ct) for ct in target_class]]=1
			
			noisevec[:,-len(listalltargetclasses1):]=torch.from_numpy(labelvec)
			noisevec=noisevec.to(device)
			all_selected_universals=ganU(noisevec)

			for c in range(len(targetclasses)):
				selected_universals=all_selected_universals[c]
				universal_comb=selected_universals.view(selected_universals.shape[0],-1)
				a=torch.linalg.vector_norm(universal_comb,ord=float('inf'),dim=1)
				tempU=universal_comb*(epsilon_norm/(a+1e-9))[:,None]
				allcombineduaps.append(torch.clone(tempU))

		
		model.apply(self.set_bn_eval)
		model.eval()
		

		totals={i:torch.Tensor([1]) for i in range(len(targetclasses))}
		success={i:torch.Tensor([0]) for i in range(len(targetclasses))}
		flipped={i:torch.Tensor([0]) for i in range(len(targetclasses))}
		perf_matches={i:torch.Tensor([0]) for i in range(len(targetclasses))}

		model.eval()
		with torch.no_grad():
			# Combine all data loaders so each batch has images for each possible target label combination. i.e., the batch would contain images for {Person, Car, Cat} and images containing these labels: {Person, Car, Dog}

			for i, dataall in enumerate(tqdm(zip(*alltestloaders))):

				for jk,t in enumerate(targetclasses):

					target_class=torch.Tensor(t).long()
					
					
					(all_inputs,img_ids),labels=dataall[jk]
					all_inputs=all_inputs.to(device)
					labels=labels.to(device).float()

					# outputs,_,_ = model(all_inputs,{'epsilon_norm':0.0})
					outputs,_,_ = model(all_inputs,None)
					
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()

					# To ensure that the model originally predicts the target classes in the images and only choose images on which model is accurately predicting the target classes.
					indices=torch.sum(1-labels[:,target_class],dim=1)==0

					if torch.count_nonzero(indices)<=1:
						continue 
					

					all_inputs=all_inputs[indices]
					labels=labels[indices]

					img_ids=np.array(img_ids)[indices.detach().cpu().numpy()]
					newlabels=torch.clone(labels)							
					#newlabels define new target labels that we would want i.e., if original predicted labels were [1 0 1] and we want to attack class 2, our new target label would be [1 0 0]
					newlabels[:,target_class] = 1 - newlabels[:,target_class]
					newlabels=torch.squeeze(newlabels,dim=0)
					
					assert(torch.sum(newlabels[:,target_class])==0)

					
					tempsucc=0
					flipped_labels=0

					model_params={
					'p_norm':p_norm,
					'epsilon_norm':epsilon_norm,
					'target_label':newlabels,
					'target_class':target_class
					}
				
					target_label=newlabels 
					
					
					
					tempU=allcombineduaps[jk]

					
					################################################################################################
					
					inputs=torch.clone(all_inputs) + tempU.view(1,all_inputs.shape[1],all_inputs.shape[2],-1)

					inputs=torch.clip(inputs,0.0,1.0)
					

					outputs,_,_ = model(inputs,model_params)

					
					outputs=torch.where(outputs>0,1,0)
					
					# To calculate ntr_rate, we first need to find images which were successfully attacked. So we compute flip_select
					flip_select=torch.sum((outputs[:,target_class]==newlabels[:,target_class]).float(),axis=1)==targetsize
					
					tempsucc=torch.count_nonzero(flip_select).cpu()
					flipped_labels=torch.count_nonzero(outputs[flip_select,:].flatten()!=newlabels[flip_select,:].flatten()).cpu().item()#-tempsucc.item()

					perf_match=outputs!=newlabels
					perf_match=torch.sum(perf_match.float(),dim=1)
					perf_match=torch.count_nonzero(perf_match==0.0)


					totals[jk]+=outputs.shape[0]
					flipped[jk]+=flipped_labels
					success[jk]+=tempsucc
					perf_matches[jk]+=perf_match.cpu().item()
					

				tempsuccess=[(success[jk]/(totals[jk]+1e-10)).item() for jk in range(len(targetclasses))]
				tempflipped=[(flipped[jk]/(success[jk]+1e-10)).item() for jk in range(len(targetclasses))]
				teststat='\nTest | \nTotals: '+', '.join(["{:.1f}".format((totals[jk]).item()) for jk in range(len(targetclasses))])+'\nPercentage: '+', '.join(["{:.2f}".format(tval) for tval in tempsuccess])+'\nFlipped: '+', '.join(["{:.2f}".format(tval) for tval in tempflipped])+'\nPerfect: '+', '.join(["{:.2f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(len(targetclasses))])
				# teststat='\nTest | \nTotals: '+', '.join(["{:.1f}".format((totals[jk]).item()) for jk in range(len(targetclasses))])+'\nPercentage: '+', '.join(["{:.2f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(len(targetclasses))])+'\nFlipped: '+', '.join(["{:.2f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(len(targetclasses))])+'\nPerfect: '+', '.join(["{:.2f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(len(targetclasses))])
				teststat +=f"\nMean Success: {np.mean(tempsuccess):3f}, Mean Flipped for target {targetsize}: {np.mean(tempflipped)/(num_classes-targetsize)}"
				print(teststat)
			

		# print(teststat)
		statstr='\n\n'+str(method_name)+'\nDataset:'+datasetname+'\nTarget size:'+str(targetsize)+'\n'+teststat
		self.logger.write(statstr)


args=parser.parse_args()
tester=Tester(args.configfile,args.expname,args.mode)
tester.test()
