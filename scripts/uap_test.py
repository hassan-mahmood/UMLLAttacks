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
		criterion=params['criterion']
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
		self.logger.write('Loss:',criterion)
        
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		
		model=model.cuda()

		#model.dotemp()

		model.eval()
		# print(model.U.shape)
		# data=model.U.data
		# np.save('/mnt/raptor/hassan/data/abc',data.cpu().numpy())
		# 0/0

		#torch.set_grad_enabled(False)
		self.logger.write('Model Evaluation:')
		#vallabels=np.zeros(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
		#valpreds=np.empty(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
		

		#alltargetlabels=list([[k] for k in range(num_classes)])
		alltargetlabels=[[0,5],[0,12],[0,11],[0,8],[0,18]]
		#alltargetlabels=[[0,14,16],[0,15,18],[0,10,11],[0,15,16]]

# 		{'0_14_16': 45, '0_15_18': 26, '0_10_11': 24, '0_5_8': 1, '0_14_15_16': 19, '0_15_16': 59, '0_15_19': 11, '15_18_19': 2
# 0, '15_16_18': 3, '0_5_11': 3, '0_11_12': 13, '15_16_17': 13, '0_15_16_18': 2, '0_14_15': 2, '15_17_18': 12, '0_4_18': 5, '0_18_19': 1, '0_8_12': 3, '14_15_16': 5, '0_8_11': 4, '0_15_17': 1, '0_15_16_17': 2, '0_14_19': 1, '0_3_5': 1}

		totals={k:{i:torch.Tensor([1]) for i in range(len(alltargetlabels))} for k in ['on','off','mixed']}
		success={k:{i:torch.Tensor([0]) for i in range(len(alltargetlabels))} for k in ['on','off','mixed']}
		flipped={k:{i:torch.Tensor([0]) for i in range(len(alltargetlabels))} for k in ['on','off','mixed']}
		perf_matches={k:{i:torch.Tensor([0]) for i in range(len(alltargetlabels))} for k in ['on','off','mixed']}

		# success={i:torch.Tensor([0]) for i in range(len(alltargetlabels))}
		# flipped={i:torch.Tensor([0]) for i in range(len(alltargetlabels))}
		# perf_matches={i:torch.Tensor([0]) for i in range(len(alltargetlabels))}

		# there will be three different types of 

		#current_target_value=0.0
		# train_labels_files=['/mnt/raptor/hassan/data/voc/vocdevkit/voc2012/ImageSets/Main/train_labels_AT.h5','/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/train_labels_AT.h5']
		# data=[]
		# for t in train_labels_files:
		# 	data.append(pd.read_hdf(t,key='df').iloc[:,:-1].to_numpy())

		# data=np.concatenate(data)
		# data=np.sum(data, axis=1)
		# print(np.where(data==0))

		# 0/0
		with torch.no_grad():
			for i,data in enumerate(tqdm(test_dataloader)):
				#img_ids,inputs, labels = data 
				(all_inputs,img_ids),labels=data
				all_inputs=all_inputs.to(device)
				labels=labels.to(device).float()
				
				outputs,features = model(all_inputs,{'epsilon_norm':0.0})
				#outputs,features = model(all_inputs,epsilon_norm=0.0,target_label=labels,target_class=torch.Tensor([0]).long())
				labels=torch.clone(outputs).detach()
				labels=torch.where(labels>0,1,0).float()
				
				
				#for jk in range(num_classes):
				for targetlabelsidx,mytargetlabels in enumerate(alltargetlabels):

					nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))
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
					# for jk in mytargetlabels:
					# 	newlabels[:,jk] = 1 - newlabels[:,jk]

					newlabels[:,mytargetlabels] = 1 - newlabels[:,mytargetlabels]

					model_params={
					'p_norm':p_norm,
					'epsilon_norm':epsilon_norm,
					'target_label':newlabels,
					'combine_uaps':combine_uaps,
					'target_class':torch.Tensor(mytargetlabels).long()#(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
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

					outputs=torch.where(outputs>0,1,0)

					on_indices=torch.ones((outputs.shape[0],),dtype=torch.bool).cuda()
					off_indices=torch.ones((outputs.shape[0],),dtype=torch.bool).cuda()
					
					for jk in mytargetlabels:
						on_indices=torch.logical_and(newlabels[:,jk]==1,on_indices)
						off_indices=torch.logical_and(newlabels[:,jk]==0,off_indices)

					mixed_indices=(1-torch.logical_or(on_indices,off_indices).long()).type(torch.bool)

					indices=[on_indices,off_indices,mixed_indices]
					for idx, idx_type in enumerate(['on','off','mixed']):
						current_indices=indices[idx]
						temptotal=torch.count_nonzero(current_indices)
						if (temptotal==0):
							continue 
						
						totals[idx_type][targetlabelsidx]+=temptotal.cpu()

						perf_match=outputs[current_indices]!=newlabels[current_indices]
						perf_match=torch.sum(perf_match.float(),dim=1)
						perf_match=torch.count_nonzero(perf_match==0.0)
					
						#flip_select=outputs[current_indices,mytargetlabels]==newlabels[current_indices,mytargetlabels]
						
						tempout=outputs[current_indices][:,mytargetlabels]==newlabels[current_indices][:,mytargetlabels]
						successindices=torch.sum(tempout.float(),dim=1)==len(mytargetlabels)
						success[idx_type][targetlabelsidx]+=torch.count_nonzero(successindices.float()).cpu().item()
						flip_select=(outputs[current_indices][successindices,:]!=newlabels[current_indices][successindices,:]).cpu().sum(dim=1)#.item()#-tempsucc.item()
						
						flipped[idx_type][targetlabelsidx]+=(flip_select/len(nontargetlabels)).sum().item()#.mean().item()
						perf_matches[idx_type][targetlabelsidx]+=torch.count_nonzero((flip_select==0).float()).item()

				
				# for idx, idx_type in enumerate(['on','off','mixed']):
				# 	print(idx_type.capitalize()+'- Total:',totals[idx_type][targetlabelsidx].item(),', Success:',success[idx_type][targetlabelsidx].item(),', Flipped:',flipped[idx_type][targetlabelsidx].item())
				#print(idx_type.capitalize()+'- Total:',totals[idx_type][targetlabelsidx].item(),', Success:',,', Flipped:',flipped[idx_type][targetlabelsidx].item()/(success[idx_type][targetlabelsidx].item()+1e-9))
				
				for idx, idx_type in enumerate(['on','off','mixed']):
					#print(idx_type.capitalize()+'- Total:',totals[idx_type][targetlabelsidx].item(),', Success:',,', Flipped:',)
					print('\nTarget:',idx_type.capitalize())
					print('Total  = [',', '.join(["{:4.2f}".format(totals[idx_type][targetlabelsidx].item()) for targetlabelsidx,mytargetlabels in enumerate(alltargetlabels)]),']')
					print('Success = [',', '.join(["{:4.2f}".format(success[idx_type][targetlabelsidx].item()/(totals[idx_type][targetlabelsidx].item()+1e-9)) for targetlabelsidx,mytargetlabels in enumerate(alltargetlabels)]),']')
					print('Flipped = [',', '.join(["{:4.2f}".format(flipped[idx_type][targetlabelsidx].item()/(success[idx_type][targetlabelsidx].item()+1e-9)) for targetlabelsidx,mytargetlabels in enumerate(alltargetlabels)]),']')


				# #print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk])).item()) for jk in range(num_classes)]))
				# print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
				# 	'\nFlipped:',', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)]),
				# 	'\nPerfect:',', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
				# 	)
				

				# #print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk])).item()) for jk in range(num_classes)]))
				# print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
				# 	'\nFlipped:',', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)]),
				# 	'\nPerfect:',', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
				# 	)
				

				#bceloss,normloss,parent_margin_loss,neg_margin_loss,loss=criterion(outputs,labels,model.get_fc_weights())
			# writer.add_scalar('Val Accuracy',val_acc,epoch)
			0/0
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

			#if(epoch%weight_store_every_epochs==0):
			if(epoch%2==0):
				store_checkpoint(checkpointdict,os.path.join(weights_dir,'model-'+str(epoch)+'.pt'))



	
args=parser.parse_args()
trainer=Tester(args.configfile,args.expname)
trainer.test()

