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
from utils.modelfactory import *
sys.path.append('ASL/')
#from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from src.models import create_model
from scipy.linalg import subspace_angles
pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/voc.ini')
parser.add_argument('--mode')
parser.add_argument('expname')
np.set_printoptions(precision=3)

torch.backends.cudnn.deterministic = True
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)



class Trainer:
	def __init__(self,configfile,experiment_name,mode):
		self.mode=mode
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':mode})	

		#self.parse_data(configfile,experiment_name,losstype)
	
	def set_bn_eval(self,module):
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()

	
	def train(self):
		
		params=self.parseddata.build()

		device=torch.device('cuda:0')
		self.logger=params['logger']
		writer=params['writer']

		# datasetname='nus'
		datasetname=params['dataset_name']
		pred_img_ids_file=ast.literal_eval(params[self.mode+'_pred_img_ids_file'])
		pred_label_file=ast.literal_eval(params[self.mode+'_pred_file'])

		num_classes=params['num_classes']
		epsilon_norm=float(ast.literal_eval(params['eps_norm']))
		p_norm=float(params['p_norm'])
		
		# epsilon_norm=0.05
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		##########################################
		weights_dir=params['weights_dir']
		weight_name='oracle_fixed_'+'eps_'+str(epsilon_norm)+'-'
		print('Weights dir and name:',os.path.join(weights_dir,weight_name))
		
		model_path=params['main_model_path']
		print(model_path)
		targetsize=int(params['targetsize'])
		target_classes_dir=params['target_classes_dir']
		datasetname=params['dataset_name']
		model_name=params['model_name']
		criterion=nn.BCEWithLogitsLoss(reduction='none')
		
		model=createmodel(model_name,model_path,self.logger,datasetname,argparse.Namespace(**params))

		# model_name='mldecoder'

		# model=createmodel(model_name,model_path,argparse.Namespace(**params))

		oracle1=torch.load(params['oracle1_weight'])['model_state']
		listalltargetclasses1=sum(pickle.load(open(os.path.join(target_classes_dir,str(1)),'rb'))['target_classes'],[])
		targetdata=pickle.load(open(os.path.join(target_classes_dir,str(targetsize)),'rb'))

		
		# if datasetname=='NUSWIDE':
		# 	# oracle1 = torch.load('/mnt/raptor/hassan/UMLL/weights/NUSWIDE/asl_oracle1_fixed_target1/oracle_fixed_orth_eps_0.05-2.pt')['model_state']#[0,:,:].unsqueeze(dim=0)
		# 	# oracle1=torch.load('/mnt/raptor/hassan/UMLL/weights/NUSWIDE/mldecoder/mldecoder_oracle1_fixed_target1/oracle_fixed_eps_0.05-6.pt')['model_state']#[0,:,:].unsqueeze(dim=0)
		# 	oracle1=torch.load('/mnt/raptor/hassan/UMLL/weights/NUSWIDE/mldecoder/mldecoder_oracle1_fixed_target1/oracle_fixed_eps_0.05-13.pt')['model_state']#[0,:,:].unsqueeze(dim=0)
		# 	listalltargetclasses1=sum(pickle.load(open('/mnt/raptor/hassan/UMLL/stores/NUSWIDE/target_classes/1','rb'))['target_classes'],[])
		# 	targetdata=pickle.load(open('/mnt/raptor/hassan/UMLL/stores/NUSWIDE/target_classes/'+str(targetsize),'rb'))
		# 	# imagesize=448
		# 	# args={'image_size':448,'model_name':'tresnet_l',
		# 	# 'sep_features':0,
		# 	# 'model_path':'/mnt/raptor/hassan/ModelWeights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		# 	# 'num_classes':81,'workers':0
		# 	# }
		# 	# args=argparse.Namespace(**args)
		# 	# state = torch.load(args.model_path, map_location='cpu')
		# 	# args.num_classes=81
		# 	# num_classes=args.num_classes
		# 	# args.do_bottleneck_head = False
		# 	# model = create_model(args).cuda()
		# 	# model.load_state_dict(state['model'], strict=True)
		# elif datasetname=='MSCOCO':
		# 	# oracle1 = torch.load('/mnt/raptor/hassan/UMLL/weights/NUSWIDE/asl_oracle1_fixed_target1/oracle_fixed_orth_eps_0.05-2.pt')['model_state']#[0,:,:].unsqueeze(dim=0)
		# 	oracle1=torch.load('/mnt/raptor/hassan/UMLL/weights/MSCOCO/asl/asl_oracle1_fixed_target1/oracle_fixed_eps_0.05-6.pt')['model_state']#[0,:,:].unsqueeze(dim=0)
		# 	listalltargetclasses1=sum(pickle.load(open('/mnt/raptor/hassan/UMLL/stores/MSCOCO/target_classes/1','rb'))['target_classes'],[])
		# 	targetdata=pickle.load(open('/mnt/raptor/hassan/UMLL/stores/MSCOCO/target_classes/'+str(targetsize),'rb'))
		# 	# imagesize=448
		# 	# args={'image_size':448,'model_name':'tresnet_l',
		# 	# 'sep_features':0,
		# 	# 'model_path':'/mnt/raptor/hassan/ModelWeights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		# 	# 'num_classes':81,'workers':0
		# 	# }
		# 	# args=argparse.Namespace(**args)
		# 	# state = torch.load(args.model_path, map_location='cpu')
		# 	# args.num_classes=81
		# 	# num_classes=args.num_classes
		# 	# args.do_bottleneck_head = False
		# 	# model = create_model(args).cuda()
		# 	# model.load_state_dict(state['model'], strict=True)

		# elif datasetname == 'oi':
		# 	0/0
		# 	oracle1 = torch.load('/mnt/raptor/hassan/UMLL/weights/oi/asl_oracle1_fixed_target1/oracle_fixed_eps_0.05-1.pt')['model_state']#[0,:,:].unsqueeze(dim=0)
		# 	listalltargetclasses1=sum(pickle.load(open('/mnt/raptor/hassan/UMLL/stores/oi/target_classes/1','rb'))['target_classes'],[])
		# 	targetdata=pickle.load(open('/mnt/raptor/hassan/UMLL/stores/oi/target_classes/'+str(targetsize),'rb'))
		# 	# imagesize=448
		# 	# args={'image_size':imagesize,'model_name':'tresnet_l',
		# 	# 'model_path':'/mnt/raptor/hassan/ModelWeights/oi/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# 	# 'num_classes':80,'workers':0
		# 	# }
		# 	# args=argparse.Namespace(**args)
		# 	# state = torch.load(args.model_path, map_location='cpu')
		# 	# args.num_classes = state['num_classes']

		# 	# args.do_bottleneck_head = True
		# 	# model = create_model(args).cuda()
		# 	# model.load_state_dict(state['model'], strict=True)
		# 	# num_classes=args.num_classes

			
		# 	##########################################

		model.cuda()
		
		targetdata['target_classes']=targetdata['target_classes']
		# alltargetclasses=sum(targetdata['target_classes'],[])

		alltrainloaders=[]
		allvalloaders=[]

		batchsize=50

		
		train_img_ids=pickle.load(open(ast.literal_eval(params['train_pred_img_ids_file']),'rb'))
		train_labels=np.load(ast.literal_eval(params['train_pred_file']))
		train_indices_dir=ast.literal_eval(params['train_indices_dir'])

		val_img_ids=pickle.load(open(ast.literal_eval(params['val_pred_img_ids_file']),'rb'))
		val_labels=np.load(ast.literal_eval(params['val_pred_file']))
		val_indices_dir=ast.literal_eval(params['val_indices_dir'])

		
		for tindex,jk in enumerate(tqdm(targetdata['target_classes'])):
			
			train_indices=np.load(os.path.join(train_indices_dir,str(targetsize),str('_'.join([str(v) for v in jk]))+'.npy'))
			
			traindatasetparams={
			'input_size':ast.literal_eval(params['image_size']),
			'images_dir':ast.literal_eval(params['train_images_dir']),
			'all_img_ids':np.array(train_img_ids)[train_indices].tolist(),
			'all_labels':train_labels[train_indices,:]
			}
			train_dataset=SelectiveDataset(traindatasetparams)
			assert(len(train_dataset)>=batchsize)
			#train_dataset2=ImageDataset(self.globalvars,'train')
			train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=int(batchsize), shuffle=True, num_workers=0)
			alltrainloaders.append(train_dataloader)


			val_indices=np.load(os.path.join(val_indices_dir,str(targetsize),str('_'.join([str(v) for v in jk]))+'.npy'))
			valdatasetparams={
			'input_size':ast.literal_eval(params['image_size']),
			'images_dir':ast.literal_eval(params['val_images_dir']),
			'all_img_ids':np.array(val_img_ids)[val_indices].tolist(),
			'all_labels':val_labels[val_indices,:]
			}
			val_dataset=SelectiveDataset(valdatasetparams)
			# assert(len(val_dataset)>=batchsize)
			#train_dataset2=ImageDataset(self.globalvars,'train')
			val_dataloader=torch.utils.data.DataLoader(val_dataset, batch_size=int(batchsize), shuffle=False, num_workers=0)
			allvalloaders.append(val_dataloader)
			# print(len(train_dataset),len(val_dataset))
		

		print('Data Loaders:',len(alltrainloaders),len(allvalloaders))
		print('Train dataloader sizes:',[len(t) for t in alltrainloaders])
		print('Val dataloader sizes:',[len(t) for t in allvalloaders])



		#############################################
		# allclassindices={}
		# origidxcooccur={}
		
		
		# for t in targetdata['target_classes']:
		# 	assert(len(t)==1)
		# 	t=t[0]
		# 	temp_indices=list(np.where(cooccur_thresh[t,:][alltargetclasses]!=0.0)[0])
		# 	#allclassindices[alltargetclasses.index(t)]=temp_indices
		# 	origidxcooccur[alltargetclasses.index(t)]=np.array(alltargetclasses)[temp_indices]

		# 	allclassindices[alltargetclasses.index(t)]=[alltargetclasses.index(k) for k in list(set(list(alltargetclasses)).difference(set([t])))]

		#############################################

		#model.dotemp()

		#train_optimizer = optim.SGD(model.parameters(), lr=float(1.0))#,weight_decay=1e-4)
		#train_optimizer = optim.Adam(model.parameters(), lr=float(1.0))#,weight_decay=1e-4)
		#scheduler = MultiStepLR(train_optimizer, milestones=[150], gamma=0.1)
		#train_optimizer = optim.SGD(model.get_params(), lr=float(0.1))#,weight_decay=1e-4)
		#model.Normalize_UAP(p_norm)

		#U = torch.nn.Parameter(torch.zeros((2,num_classes,3*224*224),dtype=torch.float32).cuda())
		########################################################################
		# Consider 2d subspaces for each universal
		# subspace dims: sub_dim
		sub_dim=1

		# v=torch.rand((len(targetdata['target_classes']),3*imagesize*imagesize),dtype=torch.float32)
		# v=torch.qr(v.T)[0].T
		# print(torch.matmul(v,v.T))
		

		# U = torch.nn.Parameter(v.unsqueeze(dim=0).cuda())


		# U = torch.nn.Parameter(torch.zeros((sub_dim,len(targetdata['target_classes']),3*imagesize*imagesize),dtype=torch.float32).cuda())
		U_coeff = torch.nn.Parameter(2*torch.rand((sub_dim,targetsize,len(targetdata['target_classes'])),dtype=torch.float32).cuda()-1)
		# U_coeff=torch.load('/mnt/raptor/hassan/UMLL/weights/NUSWIDE/mldecoder/mldecoder_oracle2_fixed_target3/oracle_fixed_eps_0.05-1.pt')['model_state']
		# U_coeff=torch.load('/mnt/raptor/hassan/UMLL/weights/NUSWIDE/mldecoder/mldecoder_oracle2_fixed_target5/oracle_fixed_eps_0.05-0.pt')['model_state']
		# U_coeff=torch.load('/mnt/raptor/hassan/UMLL/weights/oi/asl_oracle2_fixed_target5/oracle_fixed_eps_0.05-0.pt')['model_state']
		# U = torch.load('/mnt/raptor/hassan/UMLL/weights/nus/asl_oracle1_fixed_target1/oracle_fixed_orth_eps_0.05-0.pt')['model_state']
		# U = torch.nn.Parameter(torch.rand((sub_dim,len(targetdata['target_classes']),3*imagesize*imagesize),dtype=torch.float32).cuda())
		# U = torch.nn.Parameter(torch.rand((sub_dim,3*imagesize*imagesize),dtype=torch.float32).cuda())

		# Generate functions for each class that map noise vector to specfic vector in subspace
		
		

		########################################################################
		# U = torch.nn.Parameter(torch.zeros((2,len(targetdata['target_classes']),3*imagesize*imagesize),dtype=torch.float32).cuda())

		#self.U = torch.nn.Parameter(torch.zeros((2*self.num_classes,3*224*224),dtype=torch.float32).cuda())
		#U=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/t2/targetzeromodel88.pt')['model_state']
		# U=torch.load('/mnt/raptor/hassan/weights/nus/oracle1/oracle_samplefromUsubspace_eps_0.1_allOFF-2.pt')['model_state']
		U_coeff.requires_grad=True

		# train_optimizer = optim.SGD([U_coeff], lr=float(0.01))#,weight_decay=1e-4)
		train_optimizer = optim.SGD([U_coeff], lr=float(0.001))#,weight_decay=1e-4)
		
		# train_optimizer = optim.Adam([U], lr=float(0.001))
		scheduler = MultiStepLR(train_optimizer, milestones=[1,5,10,20], gamma=0.1)

		# nontargetlossscale=50
		nontargetlossscale=50
		start_epoch=0
		num_epochs=30
		# Train each uap for each epoch simultaneously

		for epoch in range(start_epoch,start_epoch+num_epochs):
			print('Epoch:',epoch)
			print('Epsilon Norm:',epsilon_norm)
			print(weight_name)
			print('non targeted scale:',nontargetlossscale)
			#current_target_value=1.0-current_target_value

			avg_meter=AverageMeter()
			
			model.eval()		# do not update BN
			model.apply(self.set_bn_eval)

			#torch.set_grad_enabled(True)
			#self.logger.write('\nModel Training:')
			

			# trainstat='\n\nTrain:'
			print(targetdata['target_classes'])
			
			totals={i:torch.Tensor([1]) for i in range(len(targetdata['target_classes']))}
			success={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
			flipped={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
			perf_matches={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
			
			# print(torch.sum(U,dim=-1))
			for jk,t in enumerate(targetdata['target_classes']):
				# if epoch==start_epoch:
				# 	if jk<10:
				# 		continue

				# if jk>3:
				# 	break
				target_class=torch.Tensor(t).long()
				target_selection_mask = torch.zeros(size=(batchsize,num_classes)).float().cuda()
				target_selection_mask[:,target_class]=1.0
				nontarget_selection_mask=1-torch.clone(target_selection_mask).cuda()
				zerotensor=torch.Tensor([0.0]).cuda()
				print('Starting:',jk)

				for i, dataall in enumerate(tqdm(alltrainloaders[jk])):
					
					# if i>5:
					# 	break 
					# get the inputs; data is a list of [image_ids, inputs, labels]

					# img_ids,inputs, labels = data 
					model.eval()
					model.zero_grad()
					(all_inputs,img_ids),_=dataall

					all_inputs=all_inputs.to(device)
					# labels=labels.to(device).float()

					with torch.no_grad():
						cleanoutputs,_ = model(all_inputs,{'epsilon_norm':0.0})
						if cleanoutputs.shape[0]<=1:
							continue
						#outputs,features = model(all_inputs,epsilon_norm=epsilon_norm,target_label=labels,target_class=torch.arange(num_classes),combine_uaps=True)
						newlabels=torch.clone(cleanoutputs)
						newlabels=torch.where(newlabels>0,1,0).float()
						newlabels[:,target_class] = 1 - newlabels[:,target_class]
						newlabels=torch.squeeze(newlabels,dim=0).to(device)

						indices=torch.sum(newlabels[:,target_class],dim=1)==0

					# indices=torch.sum(1-labels[:,target_class],dim=1)==0
					
					if len(indices)<=1:
						continue 

					all_inputs=all_inputs[indices]
					newlabels=newlabels[indices,:]
					cleanoutputs=cleanoutputs[indices,:]
					# labels=labels[indices,:]
					#newlabels=newlabels[indices]
					img_ids=np.array(img_ids)[indices.detach().cpu().numpy()]


					assert(torch.sum(newlabels[:,target_class])==0)

					

					# U.register_hook(lambda grad: torch.mul(torch.sign(grad),tempmask) * 0.002)
					
					tempsucc=0
					flipped_labels=0

					model_params={
					'p_norm':p_norm,
					'epsilon_norm':epsilon_norm,
					'target_label':newlabels,
					#'target_class':(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
					#'target_class':torch.Tensor([jk]).long()
					'target_class':target_class
					}
					
				
					target_label=newlabels 
					
					train_optimizer.zero_grad()
					

					# universal_comb=U[:,jk,:].view(-1,U.shape[-1])
					# print('Coeff shape:',U_coeff[:,:,jk].shape)
					# print('Selected vec shape:',oracle1[:,[listalltargetclasses1.index(v) for v in target_class],:].shape)
					# print(oracle1[:,[listalltargetclasses1.index(v) for v in target_class],:].shape)
					# print(U_coeff[:,:,jk])
					tempU=(U_coeff[:,:,jk].unsqueeze(dim=-1)*oracle1[:,[listalltargetclasses1.index(v) for v in target_class],:]).view(-1,oracle1.shape[-1]).sum(dim=0)
					# print('OUt:',out.shape)
					# tempU=torch.matmul(U_coeff[:,jk],oracle1[:,[listalltargetclasses1.index(v) for v in target_class],:].view(-1,oracle1.shape[-1]))

					# universal_comb=U[:,jk,:].view(-1,U.shape[-1])

					
					########################
					a=torch.linalg.vector_norm(tempU,ord=float('inf'))
					t=torch.minimum(torch.Tensor([epsilon_norm]).cuda(),a)
					tempU=tempU*(t/(a+1e-9))
					########################

					inputs=torch.clone(all_inputs) + tempU.view(-1,all_inputs.shape[1],all_inputs.shape[2],all_inputs.shape[3])

					# inputs= torch.clip(inputs,0.0,1.0)
					inputs=DifferentiableClamp.apply(inputs,0.0,1.0)

					outputs,features = model(inputs,model_params)
					
					#losses_dict,loss = criterion(outputs,newlabels)
					
					##########################################

					# Get the target loss

					targetloss=(criterion(outputs,newlabels.float())*target_selection_mask[:outputs.shape[0],:]).sum()/(torch.sum(target_selection_mask)+1e-9)
					

					# Get non-targeted loss
					# nontargetloss=nontargetlossscale*torch.sum(torch.maximum(-1*torch.tanh(torch.mul(outputs,cleanoutputs)),zerotensor)*nontarget_selection_mask[:outputs.shape[0],:])
					nontargetloss=nontargetlossscale*(torch.sum(torch.maximum(-1*torch.tanh(torch.mul(outputs,cleanoutputs)),zerotensor)*nontarget_selection_mask[:outputs.shape[0],:])/(torch.sum(nontarget_selection_mask)+1e-9))
					# /(torch.sum(nontarget_selection_mask)+1e-9)
					# print(targetloss,nontargetloss)
					##########################################


					

					# ###################################
					# tempweights=U.view(-1,U.shape[-1])/(torch.linalg.vector_norm(U.view(-1,U.shape[-1]),ord=2,dim=-1)[:,None]+1e-9)
					# # print((torch.linalg.vector_norm(tempweights,ord=2,dim=-1)))
					
					# # print(tempweights.shape)
					# # print(torch.matmul(tempweights,tempweights.T))
					# orthloss=torch.square(torch.matmul(tempweights,tempweights.T)-torch.eye(tempweights.shape[0]).cuda()).mean()#/tempweights.shape[0]
					# # print(orthloss)
					
					
					lossval=targetloss+nontargetloss


					losses_dict={
					'targetloss':targetloss,
					'nontargetloss':nontargetloss,
					'totalloss':lossval
					}
					# print(losses_dict)
					# print('Coefficients:\n',U_coeff.detach().cpu().numpy())
					# print(losses_dict)
					#losses_dict['orthloss']=orthloss 
					#losses_dict['lossval']=lossval
					# print(losses_dict)

					# 0/0
					# ###################################

					
					#print('sum before update:',torch.sum(model.U.grad))
					# print('Loss:',lossval)
					lossval.backward()
					
					# gradsums=[]

					# for mapping_temp_model in mapping_functions:
					# 	print(torch.sum(mapping_temp_model[0].weight).item())

					# print(gradsums)


					#torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

					train_optimizer.step()
					# train_optimizer2.step()
					# print(torch.sum(U,dim=-1))
					#print(model.uap_weights.flatten())
					# 
					#model.uap_weights.data=torch.nn.functional.normalize(model.uap_weights.data,p=1.0,dim=2)

					# with torch.no_grad():
					# 	model.Normalize_UAP(p_norm,epsilon_norm)

					

					# if targetonoff==1.0:
					# 	indices=torch.sum(1-outputs[:,target_class],dim=1)==0
					# 	print('After attack:',torch.count_nonzero(indices))
						
					# else:
					# 	indices=torch.sum(outputs[:,target_class],dim=1)==0
					# 	print('After attack:',torch.count_nonzero(indices))

					with torch.no_grad():
						outputs=torch.where(outputs>0,1,0)
						flip_select=torch.sum((outputs[:,target_class]==newlabels[:,target_class]).float(),axis=1)==targetsize
						
						tempsucc=torch.count_nonzero(flip_select).cpu()
						flipped_labels=torch.count_nonzero(outputs[flip_select,:].flatten()!=newlabels[flip_select,:].flatten()).cpu().item()#-tempsucc.item()

						perf_match=outputs!=newlabels
						perf_match=torch.sum(perf_match.float(),dim=1)
						perf_match=torch.count_nonzero(perf_match==0.0)

					
						for l in losses_dict.keys():
							avg_meter.update('tr_'+l,losses_dict[l].item())

						totals[jk]+=outputs.shape[0]
						flipped[jk]+=flipped_labels
						success[jk]+=tempsucc
						perf_matches[jk]+=perf_match.cpu().item()
						lossst='\n'+', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])
						# U.data=normalize_vec(U,max_norm=epsilon_norm,norm_p=p_norm)


					#print(', '.join(["{:.3f}".format(x) for x in [bceloss.item(),pw_bceloss.item(),U_sum_bceloss.item(),Up_sum_bceloss.item(),orthloss.item(),ind_loss.item(),normloss.item(),sumval.item()]]))

					#print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Perf:',perf_match.item(),', Flipped:',"{:.3f}".format(flipped_labels/(tempsucc.item()+1e-10)),", weights: {:.2f}".format(torch.sum(model.U).item()),", uap weights: {:.2f}".format(torch.sum(torch.linalg.vector_norm(model.uap_weights.data,dim=1,ord=1)).item()),", weights sum: {:.2f}".format(torch.sum(model.uap_weights.data)),lossst)
					# print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Perf:',perf_match.item(),', Flipped:',"{:.3f}".format(flipped_labels/(tempsucc.item()+1e-10)),", weights: {:.2f}".format(torch.sum(U).item()),lossst)
					
					# with torch.no_grad():
						
					# break 
				# break 
				
				checkpointdict={
					'optim_state':train_optimizer.state_dict(),
					'model_state':U_coeff,
					'mapping_functions':None,
					'target_classes':targetdata['target_classes'],
					'ending_class':jk,
					'epoch':epoch,
					'min_val_loss':0.0,
					'current_val_loss':0.0,
					'training_loss':0.0,
					'val_acc':0.0
				}
				print('Storing model')
				
				#if(epoch%weight_store_every_epochs==0):
				#if(epoch%2==0):
				store_checkpoint(checkpointdict,os.path.join(weights_dir,str(weight_name)+str(epoch)+'.pt'))

				
			
			checkpointdict={
				'optim_state':train_optimizer.state_dict(),
				'model_state':U_coeff,
				'mapping_functions':None,
				'target_classes':targetdata['target_classes'],
				'ending_class':-1,
				'epoch':epoch,
				'min_val_loss':0.0,
				'current_val_loss':0.0,
				'training_loss':0.0,
				'val_acc':0.0
			}
			print('Storing model')
			
			#if(epoch%weight_store_every_epochs==0):
			#if(epoch%2==0):
			store_checkpoint(checkpointdict,os.path.join(weights_dir,str(weight_name)+str(epoch)+'.pt'))
			

			trainstat='\n\nTrain | nontargetlossscale: '+str(nontargetlossscale)+'\nPercentage: '+', '.join(["{:.2f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(len(targetdata['target_classes']))])+'\nFlipped: '+', '.join(["{:.2f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(len(targetdata['target_classes']))])+'\nPerfect: '+', '.join(["{:.2f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(len(targetdata['target_classes']))])

		
			print(trainstat)
			#print('Target classes:',targetdata['target_classes'])
			#print('Norms:',[round(x,3) for x in torch.linalg.vector_norm(U.data,ord=float('inf'),dim=2).detach().cpu().numpy().tolist()[0]])

			# print('Norms:\n',torch.linalg.vector_norm(U.data,ord=float('inf'),dim=-1).detach().cpu().numpy())
			print('Coefficients:\n',U_coeff.detach().cpu().numpy())
			
			#print('Norms:',torch.linalg.vector_norm(model.U.data,ord=float('inf'),dim=3).detach().cpu().numpy()[:,:6,:].tolist())
			# tempU=model.U 
			# l=[]
			# for i in range(6):
			# 	for j in range(6):
			# 		l.append(np.rad2deg(subspace_angles(tempU[1,i,:,:].detach().cpu().numpy(),tempU[1,j,:,:].detach().cpu().numpy())).tolist())

			
			# #print(', '.join(["{:.3f}".format(j) for j in l]))
			# print(l)

			# u=model.get_params()
			
			# for q in range(2):
			# 	for r in range(num_classes):
			# 		np.save(os.path.join('/mnt/raptor/hassan/UAPs/stores/temp/',str(q)+'_'+str(r)+'.npy'),u[q][r].flatten().detach().cpu().numpy())
			
			# 	
			# print statistics
			#trainstatout = '\nTrain - Percentage:,'+', '.join(["{:.2f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk,_ in enumerate(targetdata['target_classes'])]) +'\nFlipped:'+', '.join(["{:.2f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk,_ in enumerate(targetdata['target_classes'])])+'\nPerfect:'+', '.join(["{:.2f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk,_ in enumerate(targetdata['target_classes'])])
			trainstatout=trainstat 

			scheduler.step()
			# scheduler2.step()
			
			model.eval()

			#torch.set_grad_enabled(False)
			# self.logger.write('Model Evaluation:')
			#vallabels=np.zeros(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			#valpreds=np.empty(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			
			with torch.no_grad():
				#valstat='\n\nValidation:'
				model.eval()
				model.zero_grad()
				totals={i:torch.Tensor([1]) for i in range(len(targetdata['target_classes']))}
				success={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
				flipped={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
				perf_matches={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}

				for jk,t in enumerate(targetdata['target_classes']):

					# if jk>3:
					# 	break
					target_class=torch.Tensor(t).long()
					target_selection_mask = torch.zeros(size=(batchsize,num_classes)).float().cuda()
					target_selection_mask[:,target_class]=1.0
					nontarget_selection_mask=1-torch.clone(target_selection_mask).cuda()
					zerotensor=torch.Tensor([0.0]).cuda()

					for i, dataall in enumerate(tqdm(allvalloaders[jk])):
						# if i>5:
						# 	break
						# img_ids,inputs, labels = data 
						
						(all_inputs,img_ids),_=dataall

						all_inputs=all_inputs.to(device)
						# labels=labels.to(device).float()

						with torch.no_grad():
							cleanoutputs,_ = model(all_inputs,{'epsilon_norm':0.0})
							if cleanoutputs.shape[0]<=1:
								continue
							#outputs,features = model(all_inputs,epsilon_norm=epsilon_norm,target_label=labels,target_class=torch.arange(num_classes),combine_uaps=True)
							newlabels=torch.clone(cleanoutputs)
							newlabels=torch.where(newlabels>0,1,0).float()
							newlabels[:,target_class] = 1 - newlabels[:,target_class]
							newlabels=torch.squeeze(newlabels,dim=0).to(device)

							indices=torch.sum(newlabels[:,target_class],dim=1)==0
						
						if len(indices)<=1:
							continue 

						all_inputs=all_inputs[indices]
						newlabels=newlabels[indices,:]
						cleanoutputs=cleanoutputs[indices,:]
						# labels=labels[indices,:]
						#newlabels=newlabels[indices]
						img_ids=np.array(img_ids)[indices.detach().cpu().numpy()]


						assert(torch.sum(newlabels[:,target_class])==0)

						# U.register_hook(lambda grad: torch.mul(torch.sign(grad),tempmask) * 0.002)
						
						

						model_params={
						'p_norm':p_norm,
						'epsilon_norm':epsilon_norm,
						'target_label':newlabels,
						#'target_class':(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
						#'target_class':torch.Tensor([jk]).long()
						'target_class':target_class
						}
						
					
						target_label=newlabels 
						
						

						# universal_comb=U[:,jk,:].view(-1,U.shape[-1])
						tempU=(U_coeff[:,:,jk].unsqueeze(dim=-1)*oracle1[:,[listalltargetclasses1.index(v) for v in target_class],:]).view(-1,oracle1.shape[-1]).sum(dim=0)
						# print('OUt:',out.shape)
						# tempU=torch.matmul(U_coeff[:,jk],oracle1[:,[listalltargetclasses1.index(v) for v in target_class],:].view(-1,oracle1.shape[-1]))

						# universal_comb=U[:,jk,:].view(-1,U.shape[-1])

						
						########################
						a=torch.linalg.vector_norm(tempU,ord=float('inf'))
						t=torch.minimum(torch.Tensor([epsilon_norm]).cuda(),a)
						tempU=tempU*(t/(a+1e-9))
						########################

						inputs=torch.clone(all_inputs) + tempU.view(-1,all_inputs.shape[1],all_inputs.shape[2],all_inputs.shape[3])

						# inputs= torch.clip(inputs,0.0,1.0)
						inputs=DifferentiableClamp.apply(inputs,0.0,1.0)

						outputs,features = model(inputs,model_params)
						
						#losses_dict,loss = criterion(outputs,newlabels)
						
						##########################################

						# Get the target loss

						targetloss=(criterion(outputs,newlabels.float())*target_selection_mask[:outputs.shape[0],:]).sum()/(torch.sum(target_selection_mask)+1e-9)
						

						# Get non-targeted loss
						nontargetloss=nontargetlossscale*torch.sum(torch.maximum(-1*torch.tanh(torch.mul(outputs,cleanoutputs)),zerotensor)*nontarget_selection_mask[:outputs.shape[0],:])/(torch.sum(nontarget_selection_mask)+1e-9)


						lossval=targetloss+nontargetloss


						losses_dict={
						'targetloss':targetloss,
						'nontargetloss':nontargetloss,
						'totalloss':lossval
						}


						with torch.no_grad():
							outputs=torch.where(outputs>0,1,0)
							flip_select=torch.sum((outputs[:,target_class]==newlabels[:,target_class]).float(),axis=1)==targetsize
							
							tempsucc=torch.count_nonzero(flip_select).cpu()
							flipped_labels=torch.count_nonzero(outputs[flip_select,:].flatten()!=newlabels[flip_select,:].flatten()).cpu().item()#-tempsucc.item()

							perf_match=outputs!=newlabels
							perf_match=torch.sum(perf_match.float(),dim=1)
							perf_match=torch.count_nonzero(perf_match==0.0)

						
							for l in losses_dict.keys():
								avg_meter.update('tr_'+l,losses_dict[l].item())

							totals[jk]+=outputs.shape[0]
							flipped[jk]+=flipped_labels
							success[jk]+=tempsucc
							perf_matches[jk]+=perf_match.cpu().item()
							lossst='\n'+', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])
							# U.data=normalize_vec(U,max_norm=epsilon_norm,norm_p=p_norm)


				# print(torch.where(newlabels[0,:]==1)[0],torch.where(outputs[0,:]>0)[0])
				
				# print('\nSelection Target:',targetonoff)
				# print('\nPercentage:',', '.join(["{:.2f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk,_ in enumerate(targetdata['target_classes'])]),
				# '\nFlipped:',', '.join(["{:.2f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk,_ in enumerate(targetdata['target_classes'])]),
				# '\nPerfect:',', '.join(["{:.2f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk,_ in enumerate(targetdata['target_classes'])]),
				# )

				valstat='\n\nValidation | \nPercentage: '+', '.join(["{:.2f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(len(targetdata['target_classes']))])+'\nFlipped: '+', '.join(["{:.2f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(len(targetdata['target_classes']))])+'\nPerfect: '+', '.join(["{:.2f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(len(targetdata['target_classes']))])
				

				# print(torch.where(newlabels[0,:]==1)[0],torch.where(outputs[0,:]>0)[0])
				# print('\nSelection Target: ',targetonoff)
				# print('\nPercentage:',', '.join(["{:.2f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk,_ in enumerate(targetdata['target_classes'])]),
				# 	'\nFlipped:',', '.join(["{:.2f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk,_ in enumerate(targetdata['target_classes'])]),
				# 	'\nPerfect:',', '.join(["{:.2f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk,_ in enumerate(targetdata['target_classes'])]),
				# 	)
				
				print(valstat)
				print('Target classes:',targetdata['target_classes'])


			now = datetime.now()
			
			#val_acc=eval_performance(valpreds,vallabels,all_class_names)
			
			statout='\n'+'-'*50+'\n'
			statout+=now.strftime("%d/%m/%Y %H:%M:%S")
			#train_loss,val_loss,newstatout=avg_meter.get_stats(epoch,writer)
			#print('Weights:',model.uap_weights.flatten())
			statout = statout + '\nEpoch: '+str(epoch)
			statout +='\nEpsilon norm: '+str(epsilon_norm)
			#statout=statout+' - '+newstatout+', Val Acc: %.3f'%(val_acc)+', Current target value: '+str(current_target_value)
			statout = statout + trainstatout
			#print('\nVal - Percentage: '+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),', Flipped:',', '.join(["{:.3f}".format((flipped[jk]/(totals[jk]*num_classes+1e-10)).item()) for jk in range(num_classes)]))
			#statout = statout +'\nVal - Percentage: '+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
			#statout = statout + '\nVal - Percentage:,'+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(len(targetdata['target_classes']))])+'\nFlipped:'+', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(len(targetdata['target_classes']))])+'\nPerfect:'+', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(len(targetdata['target_classes']))])
			statout = statout + valstat
			
			losses=avg_meter.get_values()
			

			statout+='\nTrain: '
			for l in losses.keys():
				if 'tr' in l:
					statout+=l.split('tr_')[1]+': '+str("{:.3f}".format(losses[l]))+', '

			statout+='\nVal: '

			for l in losses.keys():
				if 'val' in l:
					statout+=l.split('val_')[1]+': '+str("{:.3f}".format(losses[l]))+', '


			for l in losses.keys():
				writer.add_scalar(l,losses[l],epoch)

			#statout='\nEpoch: %d , Train loss: %.3f, Val loss: %.3f, Val Acc: %.3f, BCE: %.3f, Norm: %.3f, Parent Margin: %.3f, Neg Margin: %.3f' %(epoch, train_loss_meter.avg,val_loss_meter.avg,val_acc,bce_loss_meter.avg,norm_loss_meter.avg,parent_margin_loss_meter.avg,neg_margin_loss_meter.avg)

			self.logger.write(statout)
			# weight_name='model-'


			checkpointdict={
				'optim_state':train_optimizer.state_dict(),
				'model_state':U_coeff,
				'mapping_functions':None,
				'target_classes':targetdata['target_classes'],
				'epoch':epoch,
				'ending_class':-1,
				'min_val_loss':0.0,
				'current_val_loss':0.0,
				'training_loss':0.0,
				'val_acc':0.0
			}
			print('Storing model')
		
			#if(epoch%weight_store_every_epochs==0):
			#if(epoch%2==0):
			store_checkpoint(checkpointdict,os.path.join(weights_dir,str(weight_name)+str(epoch)+'.pt'))
			#store_checkpoint(checkpointdict,os.path.join(weights_dir,'modeljointeps0.025allOFF-'+str(epoch)+'.pt'))



	
args=parser.parse_args()
trainer=Trainer(args.configfile,args.expname,'train')
trainer.train()

