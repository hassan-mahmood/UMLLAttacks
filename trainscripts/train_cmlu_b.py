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
		datasetname=params['dataset_name']
		target_classes_dir=params['target_classes_dir']
		
		# datasetname='nus'

		pred_img_ids_file=ast.literal_eval(params[self.mode+'_pred_img_ids_file'])
		pred_label_file=ast.literal_eval(params[self.mode+'_pred_file'])


		epsilon_norm=float(ast.literal_eval(params['eps_norm']))
		p_norm=float(params['p_norm'])
		datasetname=params['dataset_name']
		model_name=params['model_name']
		model_path=params['main_model_path']
		imagesize=int(params['image_size'])
		num_classes=params['num_classes']
		batchsize = params['batch_size']
		step_size = params['step_size']
		
		# epsilon_norm=0.05
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		##########################################
		weights_dir=params['weights_dir']
		weight_name='cmlu_b_'+'eps_'+str(epsilon_norm)+'-'
		print('Weights dir and name:',os.path.join(weights_dir,weight_name))
		print('Dataset name:',datasetname)
		
		
		# datasetname='nus'
		criterion=nn.BCEWithLogitsLoss(reduction='none')
		

		
		model=createmodel(model_name,model_path,self.logger,datasetname, argparse.Namespace(**params))

		# targetsize = 1 because we will be training universal perturbations for each class.
		targetsize=1
		targetdata=pickle.load(open(os.path.join(target_classes_dir,str(targetsize)),'rb'))
		

		model.to(device)
		
		targetdata['target_classes']=targetdata['target_classes']
		alltargetclasses=sum(targetdata['target_classes'],[])

		alltrainloaders=[]
		allvalloaders=[]

	
		train_img_ids=pickle.load(open(ast.literal_eval(params['train_pred_img_ids_file']),'rb'))
		# The training label files are actually model predictions on clean images. We can directly use these predicted labels. Alternatively, we can use the model to predict labels on clean images before using them to learn the universal perturbations. In this code, we will be doing the latter so we don't really need to load the labels. Here I am just loading the prediction of first three classes.
		train_labels=np.load(ast.literal_eval(params['train_pred_file']))[:,:3]
		train_indices_dir=ast.literal_eval(params['train_indices_dir'])

		val_img_ids=pickle.load(open(ast.literal_eval(params['val_pred_img_ids_file']),'rb'))
		val_labels=np.load(ast.literal_eval(params['val_pred_file']))[:,:3]
		val_indices_dir=ast.literal_eval(params['val_indices_dir'])

		# We have training and validation data loaders for each class. We can combine all data loaders so each batch has images for each class. e.g., the batch would be: {{Images of Person}, {Images of Car}, {Images of Cat}}
		for tindex,class_k in enumerate(tqdm(targetdata['target_classes'])):
			
			
			train_indices=np.load(os.path.join(train_indices_dir,str(targetsize),str('_'.join([str(v) for v in class_k]))+'.npy'))
			
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


			val_indices=np.load(os.path.join(val_indices_dir,str(targetsize),str('_'.join([str(v) for v in class_k]))+'.npy'))
			valdatasetparams={
			'input_size':ast.literal_eval(params['image_size']),
			'images_dir':ast.literal_eval(params['val_images_dir']),
			'all_img_ids':np.array(val_img_ids)[val_indices].tolist(),
			'all_labels':val_labels[val_indices,:]
			}
			val_dataset=SelectiveDataset(valdatasetparams)
			assert(len(val_dataset)>=batchsize)
			#train_dataset2=ImageDataset(self.globalvars,'train')
			val_dataloader=torch.utils.data.DataLoader(val_dataset, batch_size=int(batchsize), shuffle=False, num_workers=0)
			allvalloaders.append(val_dataloader)
			# print(len(train_dataset),len(val_dataset))
		

		print('Data Loaders:',len(alltrainloaders),len(allvalloaders))
		print('Train dataloader sizes:',[len(t) for t in alltrainloaders])
		print('Val dataloader sizes:',[len(t) for t in allvalloaders])

		########################################################################
		# sub_dim set to 1. This was to do experiments if we learn multiple universal perturbations for a 'single' class e.g., if we want to learn three universal perturbations for a single class, we could set sub_dim = 3. If you want to do experiments with that, you would need to change the code slightly.
		sub_dim=1

		num_universals=len(targetdata['target_classes'])
		U = torch.nn.Parameter(torch.zeros((sub_dim,num_universals,3*imagesize*imagesize),dtype=torch.float32).to(device))
		U.requires_grad=True
		
		train_optimizer = optim.SGD([U], lr=float(1.0))#,weight_decay=1e-4)
		
		scheduler = MultiStepLR(train_optimizer, milestones=[1,5,10,20], gamma=0.1)

		nontargetlossscale=float(params['nontargetlossscale'])
		orthlossscale=1.0
		
		start_epoch=0
		num_epochs=30
		
		lossst=''
		zerotensor=torch.Tensor([0.0]).to(device)

		for epoch in range(start_epoch,start_epoch+num_epochs):
			print('Epoch:',epoch)
			avg_meter=AverageMeter()
			
			model.eval()		# do not update BN
			model.apply(self.set_bn_eval)
			
			totals={i:torch.Tensor([1]) for i in range(len(targetdata['target_classes']))}
			success={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
			flipped={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
			perf_matches={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
			

			for i, combined_batch in enumerate(tqdm(zip(*alltrainloaders))):
				# combined_batch will have {{Batch Data of Class 1}, {Batch Data of Class 2}, ....}

				# We can store gradients for each class and update after iterating through all classes. Alternatively, for efficiency, we can update the universal vector for each class as we iterate through all classes. In this code, we do the latter. The following 'allgrads' can be used for former setting. 
				allgrads=torch.zeros_like(U)

				alltargetlosses=[]
				allfixedlosses=[]

				U.grad=None
				
				# Iterate through each class and update universal vector for that class.
				for class_k,t in enumerate(targetdata['target_classes']):
					
					target_class=torch.Tensor(t).long()
					target_selection_mask = torch.zeros(size=(batchsize,num_classes)).float().to(device)
					target_selection_mask[:,target_class]=1.0

					model.eval()
					model.zero_grad()
					# Get batch for kth class
					(all_inputs,img_ids),labels=combined_batch[class_k]

					all_inputs=all_inputs.to(device)
					labels=labels.to(device).float()

					# We get the model predictions on clean data and choose the correctly predicted images to learn universal perturbations on.
					with torch.no_grad():
						cleanoutputs,features = model(all_inputs,None)
						labels=torch.clone(cleanoutputs).detach()
						labels=torch.where(labels>0,1,0).float()

						new_adversarial_label=torch.clone(labels)
						# Since labels are 0 or 1. If the model correctly predicts the target class (the predicted label is 1), we would want to learn a universal perturbation which changes it to 0. Therefore, we set new target labels as 1 - predictedlabels
						new_adversarial_label[:,target_class] = 1 - new_adversarial_label[:,target_class]
						new_adversarial_label=torch.squeeze(new_adversarial_label,dim=0)

					# Choose images which are predicted to contain the target class. 
					indices=torch.sum(1-labels[:,target_class],dim=1)==0
					
					if len(indices)<=1:
						continue 

					all_inputs=all_inputs[indices]
					new_adversarial_label=new_adversarial_label[indices,:]
					labels=labels[indices,:]
					cleanoutputs=cleanoutputs[indices,:]
					new_adversarial_label=new_adversarial_label[indices]
					img_ids=np.array(img_ids)[indices.detach().cpu().numpy()]


					assert(torch.sum(new_adversarial_label[:,target_class])==0)

					with torch.no_grad():
						# We have a tempmask to ensure that we only update the 'target class' column in the matrix U.
						tempmask=torch.zeros_like(U).to(device)
						tempmask[:,class_k,:]=1.0

					U.register_hook(lambda grad: torch.mul(torch.sign(grad),tempmask)*step_size)

					
					temp_success=0
					flipped_labels=0

					model_params={
					'p_norm':p_norm,
					'epsilon_norm':epsilon_norm,
					'target_label':new_adversarial_label,
					'target_class':target_class
					}
					
					target_label=new_adversarial_label 
					
					train_optimizer.zero_grad()
										
					################################################################################################
					# This code section will generate random \lambda as shown in the algorithm in the paper. 
					# For a batch_size B, we will generate B \lambda s. So we can train the model concurrently for various combinations.

					with torch.no_grad():
						# Similar to target_selection_mask, we have non_target_selection mask which chooses the non-target classes to be fixed.

						nontarget_selection_mask =torch.ones_like(target_selection_mask).to(device)

						# Generate B x C vector (which is the same as B lambdas). Note that U.shape[0] in our case is 1. 
						randcomb=torch.rand((new_adversarial_label.shape[0],U.shape[0],U.shape[1]))
						
						# To randomly sample t \tilde \mathcal{T} to set those values to 0, we generate lambdas with random 0 and 1 values and set values==1 to the random values generate above and set the rest to 0.
						random_binary=torch.randint(0,2,randcomb.shape).float()

						randcomb=torch.mul(randcomb,(1.0 - random_binary)).to(device)
						nontarget_selection_mask[:new_adversarial_label.shape[0],:][:,alltargetclasses]=random_binary.squeeze(dim=1).to(device)

						# The following two lines will ensure that \lambda_k = 0
						randcomb[:,:,class_k]=0.0
						nontarget_selection_mask[:,target_class]=0.0

						randcomb=randcomb.view(new_adversarial_label.shape[0],-1)

					
					################################################################################################
					# U \lambda:
					U_omega=torch.matmul(randcomb,U.view(-1,U.shape[-1]))

					universal_comb=U[:,class_k,:].view(-1,U.shape[-1])

					
					
					# v_k = u_k + U \lambda and project v_k to epsilon norm ball
					a=torch.linalg.vector_norm(universal_comb+U_omega,ord=float('inf'),dim=1)
					v_k=(universal_comb+U_omega)*(epsilon_norm/(a+1e-9))[:,None]

					################################################################################################

					inputs=torch.clone(all_inputs) + v_k.view(all_inputs.shape[0],all_inputs.shape[1],all_inputs.shape[2],-1)
					
					# Clip values between 0 and 1
					inputs=DifferentiableClamp.apply(inputs,0.0,1.0)

					outputs,_ = model(inputs,model_params)
					

					# Get the target loss
					targetloss=(criterion(outputs,new_adversarial_label.float())*target_selection_mask[:outputs.shape[0],:]).sum()/(torch.sum(target_selection_mask)+1e-9)

					nontargetloss=nontargetlossscale*(torch.sum(torch.maximum(-1*torch.tanh(torch.mul(outputs,cleanoutputs)),zerotensor)*nontarget_selection_mask[:outputs.shape[0],:])/(torch.sum(nontarget_selection_mask)+1e-9))
					################################################################################################
					
					tempweights=U.view(-1,U.shape[-1])/(torch.linalg.vector_norm(U.view(-1,U.shape[-1]),ord=2,dim=-1)[:,None]+1e-9)
					orthloss=orthlossscale*torch.square(torch.matmul(tempweights,tempweights.T)-torch.eye(tempweights.shape[0]).to(device)).mean()#/tempweights.shape[0]
					
					################################################################################################
					
					lossval=targetloss+nontargetloss+orthloss

					alltargetlosses.append(targetloss.item())
					allfixedlosses.append(nontargetloss.item())

					# ###################################
					lossval.backward()

					train_optimizer.step()

					with torch.no_grad():
						outputs=torch.where(outputs>0,1,0)
						flip_select=torch.sum((outputs[:,target_class]==new_adversarial_label[:,target_class]).float(),axis=1)==targetsize
						
						temp_success=torch.count_nonzero(flip_select).cpu()
						flipped_labels=torch.count_nonzero(outputs[flip_select,:].flatten()!=new_adversarial_label[flip_select,:].flatten()).cpu().item()#-temp_success.item()

						perf_match=outputs!=new_adversarial_label
						perf_match=torch.sum(perf_match.float(),dim=1)
						perf_match=torch.count_nonzero(perf_match==0.0)


						totals[class_k]+=outputs.shape[0]
						flipped[class_k]+=flipped_labels
						success[class_k]+=temp_success
						perf_matches[class_k]+=perf_match.cpu().item()

						# After update, clip the universal vectors between -epsilon_norm and +epsilon_norm
						U.data=normalize_vec(U,max_norm=epsilon_norm,norm_p=p_norm)

				
				############################################################################################
				

				# The following does not contribute to the loss. This is just for print/stat purposes. 
				with torch.no_grad():
					tempweights=U.view(-1,U.shape[-1])/(torch.linalg.vector_norm(U.view(-1,U.shape[-1]),ord=2,dim=-1)[:,None]+1e-9)
				
					orthloss=orthlossscale*torch.square(torch.matmul(tempweights,tempweights.T)-torch.eye(tempweights.shape[0]).to(device)).mean()#/tempweights.shape[0]
				
				losses_dict={
				'targetlosses':np.mean(alltargetlosses),
				'fixedlosses':np.mean(allfixedlosses),
				'orthloss':orthloss
				}

				if i%10==0:
					print(losses_dict)

				for l in losses_dict.keys():
					avg_meter.update('tr_'+l,losses_dict[l].item())

				
				lossst='\n'+', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])
				U.data=normalize_vec(U,max_norm=epsilon_norm,norm_p=p_norm)
				

			checkpointdict={
				'optim_state':train_optimizer.state_dict(),
				'model_state':U,
				'mapping_functions':None,
				'target_classes':targetdata['target_classes'],
				'epoch':epoch,
				# If you want to keep track of some metrics, you can add them here.
				'min_val_loss':0.0,
				'current_val_loss':0.0,
				'training_loss':0.0,
				'val_acc':0.0
			}
			print('Storing model')
			
			store_checkpoint(checkpointdict,os.path.join(weights_dir,str(weight_name)+str(epoch)+'.pt'))
			

			trainstat='\n\nTrain | nontargetlossscale: '+str(nontargetlossscale)+'\nPercentage: '+', '.join(["{:.2f}".format((success[class_k]/(totals[class_k]+1e-10)).item()) for class_k in range(len(targetdata['target_classes']))])+'\nFlipped: '+', '.join(["{:.2f}".format((flipped[class_k]/(success[class_k]+1e-10)).item()) for class_k in range(len(targetdata['target_classes']))])+'\nPerfect: '+', '.join(["{:.2f}".format((perf_matches[class_k]/(totals[class_k]+1e-10)).item()) for class_k in range(len(targetdata['target_classes']))])

		
			print(trainstat)
			
			print('Norms:\n',torch.linalg.vector_norm(U.data,ord=float('inf'),dim=-1).detach().cpu().numpy())
			
			trainstatout=trainstat 

			scheduler.step()
			
			
			model.eval()

			with torch.no_grad():
				
				model.eval()
				model.zero_grad()
				totals={i:torch.Tensor([1]) for i in range(len(targetdata['target_classes']))}
				success={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
				flipped={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}
				perf_matches={i:torch.Tensor([0]) for i in range(len(targetdata['target_classes']))}

				for class_k,t in enumerate(targetdata['target_classes']):
					
					target_class=torch.Tensor(t).long()
					target_selection_mask = torch.zeros(size=(batchsize,num_classes)).float().to(device)
					target_selection_mask[:,target_class]=1.0
					nontarget_selection_mask=1-torch.clone(target_selection_mask).to(device)
					zerotensor=torch.Tensor([0.0]).to(device)

					for i, combined_batch in enumerate(tqdm(allvalloaders[class_k])):
						
						(all_inputs,img_ids),labels=combined_batch

						all_inputs=all_inputs.to(device)
						labels=labels.to(device).float()

						with torch.no_grad():
							cleanoutputs,features = model(all_inputs,None)
							labels=torch.clone(cleanoutputs).detach()
							labels=torch.where(labels>0,1,0).float()

							new_adversarial_label=torch.clone(labels)
							new_adversarial_label[:,target_class] = 1 - new_adversarial_label[:,target_class]
							new_adversarial_label=torch.squeeze(new_adversarial_label,dim=0)

						indices=torch.sum(1-labels[:,target_class],dim=1)==0
						
						if len(indices)<=1:
							continue 

						all_inputs=all_inputs[indices]
						new_adversarial_label=new_adversarial_label[indices,:]
						labels=labels[indices,:]
						cleanoutputs=cleanoutputs[indices,:]
						new_adversarial_label=new_adversarial_label[indices]
						img_ids=np.array(img_ids)[indices.detach().cpu().numpy()]


						assert(torch.sum(new_adversarial_label[:,target_class])==0)

						with torch.no_grad():
							tempmask=torch.zeros_like(U).to(device)
							tempmask[:,class_k,:]=1.0

						U.register_hook(lambda grad: torch.mul(torch.sign(grad),tempmask) * 0.002)
						
						temp_success=0
						flipped_labels=0

						model_params={
						'p_norm':p_norm,
						'epsilon_norm':epsilon_norm,
						'target_label':new_adversarial_label,
						'target_class':target_class
						}

						target_label=new_adversarial_label 
						
						universal_comb=U[:,class_k,:].view(-1,U.shape[-1])
						
						################################################################################################

						inputs=torch.clone(all_inputs) + universal_comb.view(-1,all_inputs.shape[1],all_inputs.shape[2],all_inputs.shape[3])

						inputs=DifferentiableClamp.apply(inputs,0.0,1.0)

						outputs,features = model(inputs,model_params)
						
						################################################################################################

						# Get the target loss

						targetloss=(criterion(outputs,new_adversarial_label.float())*target_selection_mask[:outputs.shape[0],:]).sum()/(torch.sum(target_selection_mask)+1e-9)
						

						# Get non-targeted loss
						nontargetloss=nontargetlossscale*(torch.sum(torch.maximum(-1*torch.tanh(torch.mul(outputs,cleanoutputs)),zerotensor)*nontarget_selection_mask[:outputs.shape[0],:])/(torch.sum(nontarget_selection_mask)+1e-9))


						lossval=targetloss+nontargetloss


						losses_dict={
						'targetloss':targetloss,
						'nontargetloss':nontargetloss,
						'totalloss':lossval
						}


						with torch.no_grad():
							outputs=torch.where(outputs>0,1,0)
							flip_select=torch.sum((outputs[:,target_class]==new_adversarial_label[:,target_class]).float(),axis=1)==targetsize
							
							temp_success=torch.count_nonzero(flip_select).cpu()
							flipped_labels=torch.count_nonzero(outputs[flip_select,:].flatten()!=new_adversarial_label[flip_select,:].flatten()).cpu().item()#-temp_success.item()

							perf_match=outputs!=new_adversarial_label
							perf_match=torch.sum(perf_match.float(),dim=1)
							perf_match=torch.count_nonzero(perf_match==0.0)

						
							for l in losses_dict.keys():
								avg_meter.update('tr_'+l,losses_dict[l].item())

							totals[class_k]+=outputs.shape[0]
							flipped[class_k]+=flipped_labels
							success[class_k]+=temp_success
							perf_matches[class_k]+=perf_match.cpu().item()
							lossst='\n'+', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])
							U.data=normalize_vec(U,max_norm=epsilon_norm,norm_p=p_norm)

				valstat='\n\nValidation | \nPercentage: '+', '.join(["{:.2f}".format((success[class_k]/(totals[class_k]+1e-10)).item()) for class_k in range(len(targetdata['target_classes']))])+'\nFlipped: '+', '.join(["{:.2f}".format((flipped[class_k]/(success[class_k]+1e-10)).item()) for class_k in range(len(targetdata['target_classes']))])+'\nPerfect: '+', '.join(["{:.2f}".format((perf_matches[class_k]/(totals[class_k]+1e-10)).item()) for class_k in range(len(targetdata['target_classes']))])
				
				print(valstat)
				


			now = datetime.now()
			
			statout='\n'+'-'*50+'\n'
			statout+=now.strftime("%d/%m/%Y %H:%M:%S")
			statout = statout + '\nEpoch: '+str(epoch)
			statout +='\nEpsilon norm: '+str(epsilon_norm)
			statout = statout + trainstatout + valstat
			
			
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

			self.logger.write(statout)



	
args=parser.parse_args()
trainer=Trainer(args.configfile,args.expname,args.mode)
trainer.train()


