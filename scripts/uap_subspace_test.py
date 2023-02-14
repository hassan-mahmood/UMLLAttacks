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
pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/voc.ini')
parser.add_argument('expname')


class Tester:
	def __init__(self,configfile,experiment_name):
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'test'})		
		#self.parse_data(configfile,experiment_name,losstype)
	
	def set_bn_eval(self,module):
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()

	def test(self):
		params=self.parseddata.build()
		self.logger=params['logger']
		model=params['model']
		num_classes=params['num_classes']
		combine_uaps=params['combine_uaps']
		
		device=params['device']
		test_dataset=params['test_dataset']
		test_dataloader=params['test_dataloader']
		
		writer=params['writer']
		weights_dir=params['weights_dir']
		epsilon_norm=params['eps_norm']
		p_norm=params['p_norm']
		
		#model.base_model.load_state_dict(torch.load(self.checkpoint_load_path)['model'])
		#model.base_model.load_state_dict(torch.load('/mnt/raptor/hassan/weights/nus/asl/new.pt')['model_state'])
		#print('loaded again')
		
        
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		
		model=model.cuda()
		model.eval()
		#model.dotemp()

		avg_meter=AverageMeter()
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
			batch_img_ids=[]
			batch_inputs=[]
			batch_labels=[]

			for i,data in enumerate(tqdm(test_dataloader)):
				#img_ids,inputs, labels = data 
				(all_inputs,img_ids),labels=data
				all_inputs=all_inputs.to(device)
				labels=labels.to(device).float()
				
				outputs,features = model(all_inputs,{'epsilon_norm':0.0})
				#outputs,features = model(all_inputs,epsilon_norm=0.0,target_label=labels,target_class=torch.Tensor([0]).long())
				labels=torch.clone(outputs).detach()
				labels=torch.where(labels>0,1,0).float()

				#sindices=torch.where(torch.logical_and(labels[:,0]==0.0,torch.logical_and(labels[:,14]==0.0,labels[:,16]==0.0)).float()==1)[0]
				currenttargets=[0,15,16]
				sindices=torch.ones_like(labels[:,0],dtype=torch.bool)
				for ks in currenttargets:
					sindices=torch.logical_and(sindices,labels[:,ks]==0.0)
					#sindices=torch.where(torch.logical_and(sindices,labels[:,ks]==0.0).float()==1)[0]
				sindices=torch.where(sindices.float()==1)[0]

				if len(sindices)==0:
					continue 

				batch_inputs.append(all_inputs[sindices])
				batch_labels.append(labels[sindices])
				batch_img_ids+=np.array(img_ids)[sindices.cpu()].tolist()

				print('Length of batch labels:',len(batch_labels))
				if len(batch_labels)<2:
					continue

				all_inputs=torch.cat(batch_inputs,dim=0)

				labels=torch.cat(batch_labels,dim=0)

				#for jk in range(num_classes):
				#for mytargetclasses in [list(range(num_classes))]:
				for jk in range(0,1):
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
					#newlabels[:,0] = 1 - newlabels[:,0]
					for ks in currenttargets:
						newlabels[:,ks] = 1 - newlabels[:,ks]
					#newlabels[:,14] = 1 - newlabels[:,14]
					#newlabels[:,16] = 1 - newlabels[:,16]

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
					#losses_dict,loss = criterion(outputs,newlabels,[jk],use_selection_mask=True,model=model)
					
					
					#lossst=', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])

					# for l in losses_dict.keys():
					# 	avg_meter.update('test_'+l,losses_dict[l].item())

					# avg_meter.update('test_loss',loss.item())

					
					outputs=torch.where(outputs>0,1,0)
					# print(outputs[:50,:])
					# print(newlabels[:50,:])
					# 0/0
					# print((outputs-newlabels)[:40,10:])
					# 0/0
					#print(newlabels[:100,:][:,[0,5]])
					#print(outputs[:100,:][:,[0,5]])


					flip_select=torch.ones_like(outputs[:,jk],dtype=torch.bool)
					
					for p in currenttargets:
						flip_select=torch.logical_and(outputs[:,p]==newlabels[:,p],flip_select)

					tempsucc=torch.count_nonzero(flip_select).cpu()

					flipped_labels=torch.count_nonzero(outputs[flip_select,:].flatten()!=newlabels[flip_select,:].flatten()).cpu().item()#-tempsucc.item()

					perf_match=outputs!=newlabels
					perf_match=torch.sum(perf_match.float(),dim=1)
					perf_match=torch.count_nonzero(perf_match==0.0)

					
					totals[jk]+=outputs.shape[0]
					flipped[jk]+=flipped_labels
					success[jk]+=tempsucc
					perf_matches[jk]+=perf_match.cpu().item()

					#print(', '.join(["{:.3f}".format(x) for x in [bceloss.item(),pw_bceloss.item(),U_sum_bceloss.item(),Up_sum_bceloss.item(),orthloss.item(),ind_loss.item(),normloss.item(),sumval.item()]]))

					
					

					#print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Perf:',perf_match.item(),', Flipped:',"{:.3f}".format(flipped_labels/(tempsucc.item()+1e-10)),", weights: {:.2f}".format(torch.sum(model.U).item()),", uap weights: {:.2f}".format(torch.sum(torch.linalg.vector_norm(model.uap_weights.data,dim=1,ord=1)).item()),", weights sum: {:.2f}".format(torch.sum(model.uap_weights.data)),lossst)
					print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Perf:',perf_match.item(),', Flipped:',"{:.3f}".format(flipped_labels/(tempsucc.item()+1e-10)),", weights: {:.2f}".format(torch.sum(model.U).item()))


					
					#print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Flipped:',flipped_labels/num_classes,torch.sum(model.U).item())

			
				
				#print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk])).item()) for jk in range(num_classes)]))
				print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
					'\nFlipped:',', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)]),
					'\nPerfect:',', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
					)
				
				batch_img_ids=[]
				batch_inputs=[]
				batch_labels=[]

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
			statout = statout + '\nTest - Percentage:,'+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])+'\nFlipped:'+', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)])+'\nPerfect:'+', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
			
			losses=avg_meter.get_values()
			
			statout+='\nTest: '

			for l in losses.keys():
				if 'test' in l:
					statout+=l.split('test_')[1]+': '+str("{:.3f}".format(losses[l]))+', '


			for l in losses.keys():
				writer.add_scalar(l,losses[l],epoch)

			#statout='\nEpoch: %d , Train loss: %.3f, Val loss: %.3f, Val Acc: %.3f, BCE: %.3f, Norm: %.3f, Parent Margin: %.3f, Neg Margin: %.3f' %(epoch, train_loss_meter.avg,val_loss_meter.avg,val_acc,bce_loss_meter.avg,norm_loss_meter.avg,parent_margin_loss_meter.avg,neg_margin_loss_meter.avg)

			self.logger.write(statout)
			



	
args=parser.parse_args()
trainer=Tester(args.configfile,args.expname)
trainer.test()

