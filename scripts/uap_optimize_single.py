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
pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/voc.ini')
parser.add_argument('expname')


class Trainer:
	def __init__(self,configfile,experiment_name):
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'train'})		
		#self.parse_data(configfile,experiment_name,losstype)
	
	def set_bn_eval(self,module):
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()

	def train(self):
		params=self.parseddata.build()
		self.logger=params['logger']
		model=params['model']
		criterion=params['criterion']
		num_classes=params['num_classes']
		combine_uaps=params['combine_uaps']
		
		device=params['device']
		train_optimizer=params['optimizer']
		scheduler=params['scheduler']
		train_dataloader=params['train_dataloader']
		val_dataset=params['val_dataset']
		val_dataloader=params['val_dataloader']
		
		start_epoch=params['start_epoch']
		num_epochs=params['num_epochs']
		weight_store_every_epochs=params['weight_store_every_epochs']
		writer=params['writer']
		min_val_loss=params['min_val_loss']
		weights_dir=params['weights_dir']
		epsilon_norm=params['eps_norm']
		p_norm=params['p_norm']
		
		#model.base_model.load_state_dict(torch.load(self.checkpoint_load_path)['model'])
		#model.base_model.load_state_dict(torch.load('/mnt/raptor/hassan/weights/nus/asl/new.pt')['model_state'])
		#print('loaded again')
		self.logger.write('Loss:',criterion)
        
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		
		model=model.cuda()

		#model.dotemp()

		train_optimizer = optim.SGD(model.parameters(), lr=float(1.0))#,weight_decay=1e-4)
		#train_optimizer = optim.Adam(model.parameters(), lr=float(1.0))#,weight_decay=1e-4)
		scheduler = MultiStepLR(train_optimizer, milestones=[150], gamma=0.1)
		#train_optimizer = optim.SGD(model.get_params(), lr=float(0.1))#,weight_decay=1e-4)
		#model.Normalize_UAP(p_norm)
		#start_epoch=0
		num_epochs=100
		for epoch in range(start_epoch,start_epoch+num_epochs):
			#current_target_value=1.0-current_target_value

			avg_meter=AverageMeter()

 
			model.train()
			model.apply(self.set_bn_eval)
			model.model.eval()		# do not update BN

			#torch.set_grad_enabled(True)
			#self.logger.write('\nModel Training:')
			totals={i:torch.Tensor([1]) for i in range(num_classes)}
			success={i:torch.Tensor([0]) for i in range(num_classes)}
			flipped={i:torch.Tensor([0]) for i in range(num_classes)}
			perf_matches={i:torch.Tensor([0]) for i in range(num_classes)}

			for i, data in enumerate(tqdm(train_dataloader)):
				
				# get the inputs; data is a list of [image_ids, inputs, labels]
				#img_ids,inputs, labels = data 
				(all_inputs,img_ids),labels=data
				
				all_inputs=all_inputs.to(device)
				labels=labels.to(device).float()


				with torch.no_grad():
					outputs,features = model(all_inputs,{'epsilon_norm':0.0})
					#outputs,features = model(all_inputs,epsilon_norm=epsilon_norm,target_label=labels,target_class=torch.arange(num_classes),combine_uaps=True)
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()


				#for jk in range(num_classes):
				for jk in range(1,2):
					print('Class:',jk)
					for to_use_selection_mask in [False]:
					# for totargetchange in [0.0,1.0]:
						#for jk in range(1):
					#for jk in range(1,2):
						
						newlabels=torch.clone(labels)

						#select_indices=torch.where(torch.logical_and((labels[:,jk]==current_target_value),(labels[:,2]==1)))[0]
						# select_indices=torch.where(labels[:,jk]==0.0)[0]
						# if(select_indices.shape[0]==0):
						# 	continue
						
						#newlabels=torch.clone(labels[select_indices,:])
						
						#inputs=torch.clone(all_inputs[select_indices,:,:,:])
						########### experiment part #############
						#inputs= torch.rand_like(inputs)
						#newlabels=torch.zeros_like(newlabels)
						########### experiment part #############

						#if(current_target_value==0.0):
						# print(model.U[0,:5,:5])

						# newlabels[:,jk] = 1 - newlabels[:,jk]
						newlabels = torch.ones_like(newlabels)#1 - newlabels[:,jk]

						# gradmask=torch.zeros_like(model.U,dtype=torch.float32)
						# gradmask[:,jk,:]=1.0
						# model.U.register_hook(lambda grad: torch.sign(grad) * 0.1*gradmask)
						# with torch.no_grad():
						# 	tempmask=torch.zeros((2,num_classes,3*224*224),dtype=torch.float32).cuda()
						# 	tempmask[:,jk,:,:]=1.0

						# model.U.register_hook(lambda grad: torch.mul(torch.sign(grad),tempmask) * 0.002)
						model.U.register_hook(lambda grad: torch.sign(grad) * 0.002)
						#model.U.register_hook(lambda grad: grad * 0.002)
						#model.U.register_hook(lambda grad: (grad/(torch.linalg.vector_norm(grad,ord=float('inf'),dim=2,keepdim=True)+1e-10)) * 0.002)
						#model.x.register_hook(lambda grad: grad * 0.0002)
												
						#model.uap_weights.register_hook(lambda grad: grad * 0.00001)

						#newlabels[:,jk]=totargetchange

						tempsucc=0
						flipped_labels=0

						model_params={
						'p_norm':p_norm,
						'epsilon_norm':epsilon_norm,
						'target_label':newlabels,
						'combine_uaps':combine_uaps,
						#'target_class':(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
						'target_class':torch.Tensor([jk]).long()
						}
						
						train_optimizer.zero_grad()
						#inputs=torch.clone(all_inputs[select_indices,:,:,:])
						inputs=torch.clone(all_inputs)
						outputs,features = model(inputs,model_params)
						
						losses_dict,loss = criterion(outputs,newlabels,[jk],use_selection_mask=to_use_selection_mask,model=model,getfull=False)
						#print('sum before update:',torch.sum(model.U.grad))
						
						loss.backward()
						
						#torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

						train_optimizer.step()
						
						#print(model.uap_weights.flatten())
						# 
						#model.uap_weights.data=torch.nn.functional.normalize(model.uap_weights.data,p=1.0,dim=2)

						# with torch.no_grad():
						# 	model.Normalize_UAP(p_norm,epsilon_norm)

						outputs=torch.where(outputs>0,1,0)
						flip_select=outputs[:,jk]==newlabels[:,jk]
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

						#print(', '.join(["{:.3f}".format(x) for x in [bceloss.item(),pw_bceloss.item(),U_sum_bceloss.item(),Up_sum_bceloss.item(),orthloss.item(),ind_loss.item(),normloss.item(),sumval.item()]]))

						
						lossst='\n'+', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])

						#print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Perf:',perf_match.item(),', Flipped:',"{:.3f}".format(flipped_labels/(tempsucc.item()+1e-10)),", weights: {:.2f}".format(torch.sum(model.U).item()),", uap weights: {:.2f}".format(torch.sum(torch.linalg.vector_norm(model.uap_weights.data,dim=1,ord=1)).item()),", weights sum: {:.2f}".format(torch.sum(model.uap_weights.data)),lossst)
						print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Perf:',perf_match.item(),', Flipped:',"{:.3f}".format(flipped_labels/(tempsucc.item()+1e-10)),", weights: {:.2f}".format(torch.sum(model.U).item()),lossst)
						

						#print(torch.sum(model.U,dim=2))
						#print(loss)
						#Q = model.U[0]
						#R=model.U[1]

						#print('Dist weight:',torch.sum(torch.matmul(Q, Q.T)),torch.sum(torch.matmul(R, R.T)))
						
						avg_meter.update('tr_train_loss',loss.item())
						
						
						if(tempsucc==inputs.shape[0]):
						 	break
						
						# totals[jk]+=outputs.shape[0]
						# flipped[jk]+=flipped_labels
						# success[jk]+=tempsucc
						# perf_matches[jk]+=perf_match

						#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.arange(num_classes))
						#percents.append((torch.count_nonzero(outputs[:,jk]<=0)/outputs.shape[0]).item())
						
						# for p in model.parameters():
						# 	if(p.requires_grad is True):
						# 		#print(p.shape)
						# 		p.grad=0.0001*torch.sign(p.grad)
						#nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
						
						#scheduler.step()

				

				with torch.no_grad():
					model.Normalize_UAP(p_norm,epsilon_norm)
				# print(torch.where(newlabels[0,:]==1)[0],torch.where(outputs[0,:]>0)[0])
				print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
					'\nFlipped:',', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)]),
					'\nPerfect:',', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
					)
				print('Norms:',torch.linalg.vector_norm(model.U.data,ord=float('inf'),dim=2).detach().cpu().numpy().tolist())
				
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
			trainstatout = '\nTrain - Percentage:,'+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]) +'\nFlipped:'+', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)])+'\nPerfect:'+', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])

			scheduler.step()
			#losses=avg_meter.get_values()

			# lossst=''
			# for l in losses.keys():
			# 	if 'tr' in l:
			# 		lossst+=l+':'+str(losses[l])+', '
			
			
			
			model.eval()

			#torch.set_grad_enabled(False)
			self.logger.write('Model Evaluation:')
			#vallabels=np.zeros(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			#valpreds=np.empty(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			
			totals={i:torch.Tensor([1]) for i in range(num_classes)}
			success={i:torch.Tensor([0]) for i in range(num_classes)}
			flipped={i:torch.Tensor([0]) for i in range(num_classes)}
			perf_matches={i:torch.Tensor([0]) for i in range(num_classes)}

			#current_target_value=0.0
			with torch.no_grad():
				for i,data in enumerate(tqdm(val_dataloader)):
					#img_ids,inputs, labels = data 
					(all_inputs,img_ids),labels=data
					all_inputs=all_inputs.to(device)
					labels=labels.to(device).float()
					
					outputs,features = model(all_inputs,{'epsilon_norm':0.0})
					#outputs,features = model(all_inputs,epsilon_norm=0.0,target_label=labels,target_class=torch.Tensor([0]).long())
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()
					
					
					#for jk in range(num_classes):
					for jk in range(1,2):
					#for jk in range(6):
					#for jk in range(1,2):
						#for totargetchange in [0.0,1.0]:
					#for jk in range(1):
					#for jk in range(1,2):
						#select_indices=torch.where(labels[:,jk]==current_target_value)[0]
						#select_indices=torch.where(torch.logical_and((labels[:,jk]==current_target_value),(labels[:,2]==1)))[0]
						#print(torch.count_nonzero(select_indices))
						# if(torch.count_nonzero(select_indices)==0):
						# 	continue
					
						#newlabels=torch.clone(labels[select_indices,:])
						#inputs=all_inputs[select_indices,:,:,:]
						newlabels=torch.clone(labels)
						inputs=torch.clone(all_inputs)

						########### experiment part #############
						#inputs= torch.rand_like(inputs)
						#newlabels=torch.zeros_like(newlabels)
						########### experiment part #############

						#
						#newlabels = 1 - newlabels
						#newlabels[:,jk] = 1 - newlabels[:,jk]
						newlabels = torch.ones_like(newlabels)
						
						model_params={
						'p_norm':p_norm,
						'epsilon_norm':epsilon_norm,
						'target_label':newlabels,
						'combine_uaps':combine_uaps,
						'target_class':torch.Tensor([jk]).long()#(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
						}
						
						outputs,features = model(inputs,model_params)
						#newlabels[:,jk] = totargetchange
						
						#print(inputs.shape,newlabels.shape)
					
						#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=jk)
						#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.arange(num_classes))
						
						#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.Tensor([jk]).long())
						#print(torch.where(outputs[0,:]>0,1,0))
						

						# if(current_target_value==1.0):
						# 	success[jk]+=torch.count_nonzero(outputs[:,jk]<=0).cpu()
						# else:
						# 	success[jk]+=torch.count_nonzero(outputs[:,jk]>0).cpu()	
						losses_dict,loss = criterion(outputs,newlabels,[jk],use_selection_mask=False,model=model)
						
						
						lossst=', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])

						for l in losses_dict.keys():
							avg_meter.update('val_'+l,losses_dict[l].item())

						avg_meter.update('val_val_loss',loss.item())

						outputs=torch.where(outputs>0,1,0)
						flip_select=outputs[:,jk]==newlabels[:,jk]
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
						
						
						#print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Flipped:',flipped_labels/num_classes,torch.sum(model.U).item())
						print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Perf:',perf_match.item(),', Flipped:',"{:.3f}".format(flipped_labels/(tempsucc.item()+1e-10)),", weights: {:.2f}".format(torch.sum(model.U).item()),lossst)

				
					
					#print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk])).item()) for jk in range(num_classes)]))
					print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
						'\nFlipped:',', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)]),
						'\nPerfect:',', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
						)
					

				#bceloss,normloss,parent_margin_loss,neg_margin_loss,loss=criterion(outputs,labels,model.get_fc_weights())
			# writer.add_scalar('Val Accuracy',val_acc,epoch)

			now = datetime.now()
			
			#val_acc=eval_performance(valpreds,vallabels,all_class_names)
			statout=now.strftime("%d/%m/%Y %H:%M:%S")
			#train_loss,val_loss,newstatout=avg_meter.get_stats(epoch,writer)
			#print('Weights:',model.uap_weights.flatten())
			statout = statout + '\nEpoch: '+str(epoch)+'\n'
			#statout=statout+' - '+newstatout+', Val Acc: %.3f'%(val_acc)+', Current target value: '+str(current_target_value)
			statout = statout + trainstatout
			#print('\nVal - Percentage: '+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),', Flipped:',', '.join(["{:.3f}".format((flipped[jk]/(totals[jk]*num_classes+1e-10)).item()) for jk in range(num_classes)]))
			#statout = statout +'\nVal - Percentage: '+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
			statout = statout + '\nVal - Percentage:,'+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])+'\nFlipped:'+', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)])+'\nPerfect:'+', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
			
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
			weight_name='model-'

			checkpointdict={
				'optim_state':train_optimizer.state_dict(),
				'model_state':model.state_dict(),
				'epoch':epoch,
				'min_val_loss':0.0,
				'current_val_loss':0.0,
				'training_loss':0.0,
				'val_acc':0.0
			}
			print('Storing model')
			#if(epoch%weight_store_every_epochs==0):
			#if(epoch%2==0):
			store_checkpoint(checkpointdict,os.path.join(weights_dir,'model-'+str(epoch)+'.pt'))



	
args=parser.parse_args()
trainer=Trainer(args.configfile,args.expname)
trainer.train()

