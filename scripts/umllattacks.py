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
from Attacks.mll import * 

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

		model.dotemp()

		# train_optimizer = optim.SGD(model.parameters(), lr=float(1.0))#,weight_decay=1e-4)
		# scheduler = MultiStepLR(train_optimizer, milestones=[5,20], gamma=0.1)

		orthmll=OrthMLLAttacks()

		#pert_param = torch.nn.Parameter(torch.zeros((2,self.num_classes,3*224*224),dtype=torch.float32).cuda())
		#pert_param.requires_grad=True

		train_optimizer = torch.optim.Adam(model.parameters(), lr=float(1e-2))#,weight_decay=1e-4)
		#train_optimizer = optim.SGD(model.parameters(), lr=float(1e-2)#,weight_decay=1e-4)
		scheduler = MultiStepLR(train_optimizer, milestones=[5], gamma=0.1)


		#train_optimizer = optim.SGD(model.get_params(), lr=float(0.1))#,weight_decay=1e-4)
		#model.Normalize_UAP(p_norm)
		#start_epoch=0
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


				for jk in range(num_classes):
				#for jk in range(1,2):
					print('Class:',jk)
					for to_use_selection_mask in [True]:
					# for totargetchange in [0.0,1.0]:
						#for jk in range(1):
					#for jk in range(1,2):
						
						newlabels=torch.clone(labels)

						newlabels[:,jk] = 1 - newlabels[:,jk]

						# gradmask=torch.zeros_like(model.U,dtype=torch.float32)
						# gradmask[:,jk,:]=1.0
						# model.U.register_hook(lambda grad: torch.sign(grad) * 0.1*gradmask)
						
						#model.U.register_hook(lambda grad: torch.sign(grad) * 0.001)
						
						#model.uap_weights.register_hook(lambda grad: grad * 0.00001)

						#newlabels[:,jk]=totargetchange

						tempsucc=0
						flipped_labels=0
						mytargetlabels=[jk]

						model_params={
						'p_norm':p_norm,
						'epsilon_norm':epsilon_norm,
						'target_label':newlabels,
						'combine_uaps':combine_uaps,
						#'target_class':(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
						'target_class':torch.Tensor(mytargetlabels).long()
						}
						
						train_optimizer.zero_grad()
						#inputs=torch.clone(all_inputs[select_indices,:,:,:])
						inputs=torch.clone(all_inputs)
						outputs,features = model(inputs,model_params)
						
						#losses_dict,loss = criterion(outputs,newlabels,[jk],use_selection_mask=to_use_selection_mask,model=model)
						###################

						#tempinputs=torch.clamp(torch.clone(inputs+pert_param).detach(),0.0,1.0)
						#tempinputs=inputs 
						#best_e=orthmll.UMLLperturb(model,tempinputs,newlabels,mytargetlabels,nontargetlabels,eps_val,norm,orth_step_size ,criterion, out_iterations,in_iterations)
						#best_e=best_e.view(3,224,224)
						
						nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))
						#losses_dict,loss = criterion(outputs,newlabels,[jk],use_selection_mask=to_use_selection_mask,model=model)
						target_grad,nontarget_grad,losses_dict=orthmll.UMLLperturb(model,inputs,newlabels,mytargetlabels,nontargetlabels,criterion,model_params,to_use_selection_mask)
						loss=losses_dict['bce_loss']
						#print(target_grad.shape)
						gradsval=torch.zeros_like(model.U)
						gradsval[:,jk,:]=-1*target_grad
						model.U.grad=gradsval
						#print(model.U[:,mytargetlabels,:].shape)
						#0/0
						#print(target_grad.shape)
						#target_grad=torch.sum(target_grad.squeeze(),dim=0)
						#pert_param.grad=-1*target_grad.view(3,224,224)
						#pert_param.requires_grad=True 
						#loss = torch.square(torch.matmul(target_grad.squeeze(),pert_param.view(-1))) #+ torch.mul()
						#loss=loss.sum()
						#loss.backward()

						best_e=torch.clone(target_grad).detach()
						#print('Sum grad:',torch.sum(best_e))
						# print(loss.shape)(20)

						# loss=torch.matmul(torch.transpose(nontarget_grad,1,2),pert_param.view(-1))
						# print(loss.shape) 20 x 19
						#0/0
						#pert_param.grad=best_e

						#print('Grad sum:',torch.sum(model.U.grad))
						train_optimizer.step()
						
						#print(model.uap_weights.flatten())
						# 
						#model.uap_weights.data=torch.nn.functional.normalize(model.uap_weights.data,p=1.0,dim=2)

						with torch.no_grad():
							model.Normalize_UAP(p_norm,epsilon_norm)

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



				# print(torch.where(newlabels[0,:]==1)[0],torch.where(outputs[0,:]>0)[0])
				print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
					'\nFlipped:',', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)]),
					'\nPerfect:',', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
					)
				print('Norms:',torch.linalg.vector_norm(model.U.data,ord=float('inf'),dim=2).detach().cpu().numpy().tolist())
				
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
			
			# print('\n',lossst)

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
					
					
					for jk in range(num_classes):
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
						newlabels[:,jk] = 1 - newlabels[:,jk]

						mytargetlabels=[jk]

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
						losses_dict,loss = criterion(outputs,newlabels,mytargetlabels,use_selection_mask=True,model=model)
						
						
						lossst=', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])

						for l in losses_dict.keys():
							avg_meter.update('val_'+l,losses_dict[l].item())

						avg_meter.update('val_val_loss',loss.item())

						outputs=torch.where(outputs>0,1,0)
						
						perf_match=outputs!=newlabels
						perf_match=torch.sum(perf_match.float(),dim=1)
						perf_match=torch.count_nonzero(perf_match==0.0)
						
						

						totals[jk]+=outputs.shape[0]
						flip_select=outputs[:,jk]==newlabels[:,jk]
						tempsucc=torch.count_nonzero(flip_select).cpu()
						success[jk]+=tempsucc
						flipped[jk]+=torch.count_nonzero(outputs[flip_select,:].flatten()!=newlabels[flip_select,:].flatten()).cpu().item()#-tempsucc.item()
						perf_matches[jk]+=perf_match.cpu().item()
						
						
						print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Flipped:',flipped_labels/num_classes,torch.sum(model.U).item())

				
					
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

			#if(epoch%weight_store_every_epochs==0):
			store_checkpoint(checkpointdict,os.path.join(weights_dir,'model-'+str(epoch)+'.pt'))



	
args=parser.parse_args()
trainer=Trainer(args.configfile,args.expname)
trainer.train()




# import os
# import sys
# sys.path.append('./')
# from utils.utility import *
# from utils.confparser import *
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import torchvision
# #from torch.autograd.functional import jacobian
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import copy
# from tqdm import tqdm 
# from Models import *
# #from Datasets.NUSDataset import NUSImageDataset as ImageDataset
# from Datasets import * 
# import re
# from Logger.Logger import *
# import configparser
# import ast, json
# import argparse
# import pickle
# from datetime import datetime
# from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
# from Attacks.apgd import *
# from Attacks.mll import * 
# from Attacks.projected_gradient_descent import *
# import random 
# import gc 
# import time
# import itertools 
# #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

# pickle.HIGHEST_PROTOCOL = 4
# torch.set_printoptions(edgeitems=27)
# parser=argparse.ArgumentParser()
# parser.add_argument('--configfile',default='configs/voc.ini')
# parser.add_argument('expname')

# torch.backends.cudnn.deterministic = True
# np.random.seed(999)
# random.seed(999)
# torch.manual_seed(999)

# # import collections
# # data=pd.read_hdf('/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/test_labels_AT.h5',key='df').iloc[:,:-1].to_numpy()
# # freq={}
# # for i in range(data.shape[1]):
# # 	#print(collections.Counter(data[:,i]))
# # 	temp=collections.Counter(data[:,i])
# # 	freq[i]=temp[1]
	

# # print(freq)
# # 0/0

# # from sympy import Matrix
# # from sympy.physics.quantum import TensorProduct
# # def my_nullspace(At, rcond=None):

# #     ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)

# #     #At[2,:]=At[1,:]*2
# #     #ut, st, vht = torch.linalg.svd(At, full_matrices=True)
# #     print(st)
# #     print(st.shape)
# #     print(ut.shape,vht.shape)


# #     #At = Matrix(At.detach().numpy())
# #     #nspace=At.nullspace()

# #     #print(TensorProduct(nspace,At))
# #     #print(nspace[0])
# #     print(At.shape)
# #     print(torch.matmul(vht,At))
# #     print(torch.matmul(uht,At))

# #     0/0

# #     vht=vht[st.shape[0]:,:]
# #     print(torch.matmul(vht,At.t()))
# #     0/0
# #     vht=vht.T        

# #     print(torch.matmul(vht.t(),At.t()))
# #     0/0
# #     Mt, Nt = ut.shape[0], vht.shape[1] 
# #     if rcond is None:

# #         rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
# #     tolt = torch.max(st) * rcondt
# #     numt= torch.sum(st > tolt, dtype=int)
# #     nullspace = vht[numt:,:].T.cpu().conj()
# #     # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
# #     return nullspace

# # G = torch.randn((10,3),dtype=torch.float32)
# # G_orth = torch.qr(G)[0].t()

# # n=my_nullspace(G_orth)
# # # print(n.shape)
# # # print(torch.matmul(G_orth,n))
# # print(G.shape,G_orth.shape)
# # print(torch.matmul(G_orth,G))



# # a=pd.read_hdf(os.path.join('/mnt/raptor/hassan/UAPs/best_7_10_13/','best_mll_0.h5'),key='df',mode='r')
# # a=a.iloc[:,:-1].to_numpy()
# # print(a)
# # 0/0
# # 0/0
# ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'dining_table', 'pot_plant', 'sofa', 'tv_monitor']
# #print(pickle.load(open('/mnt/raptor/hassan/UAPs/KG_data/voc/class_names','rb')))
# #0/0
# #bird 1, Dog 4, bicycle 8, chair 15

# class Trainer:
# 	def __init__(self,configfile,experiment_name):
# 		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'train'})		
# 		#self.parse_data(configfile,experiment_name,losstype)
	
# 	def set_bn_eval(self,module):
# 		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
# 			module.eval()

# 	def compute_apgd(self,model,inputs,newlabels,criterion,selection_mask_target):
# 		apgdt = APGDAttack(model, eps=0.05, norm='Linf', n_iter=400, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
# 		return apgdt.perturb(inputs,newlabels,criterion,use_target_indices=False,selection_mask=selection_mask_target)


# 	def train(self):
# 		params=self.parseddata.build()
# 		self.logger=params['logger']
# 		model=params['model']
# 		criterion=params['criterion']
# 		num_classes=params['num_classes']
# 		combine_uaps=params['combine_uaps']
		
# 		device=params['device']
# 		train_optimizer=params['optimizer']
# 		scheduler=params['scheduler']
# 		train_dataloader=params['train_dataloader']
# 		val_dataset=params['val_dataset']
# 		val_dataloader=params['val_dataloader']
		
# 		start_epoch=params['start_epoch']
# 		num_epochs=params['num_epochs']
# 		weight_store_every_epochs=params['weight_store_every_epochs']
# 		writer=params['writer']
# 		min_val_loss=params['min_val_loss']
# 		weights_dir=params['weights_dir']
# 		epsilon_norm=params['eps_norm']
# 		p_norm=params['p_norm']
		
# 		#model.base_model.load_state_dict(torch.load(self.checkpoint_load_path)['model'])
# 		#model.base_model.load_state_dict(torch.load('/mnt/raptor/hassan/weights/nus/asl/new.pt')['model_state'])
# 		#print('loaded again')
# 		self.logger.write('Loss:',criterion)
		
        
		
# 		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
# 		model=model.cuda()

# 		#model.dotemp()

# 		#train_optimizer = optim.SGD(model.parameters(), lr=float(1.0))#,weight_decay=1e-4)
# 		#train_optimizer = optim.SGD([self.U], lr=float(0.01))#,weight_decay=1e-4)
# 		mstones=[5,10,20,40,60]
# 		scheduler = MultiStepLR(train_optimizer, milestones=mstones, gamma=0.1)
# 		current_target_value=0.0
# 		#train_optimizer = optim.SGD(model.get_params(), lr=float(0.1))#,weight_decay=1e-4)
# 		#model.Normalize_UAP(p_norm)
# 		#start_epoch=0
# 		# lrval=0.1
# 		# eps_val=0.05
# 		# num_iterations=400
# 		# alpha=eps_val/float(num_iterations)
# 		eps_val=0.005#2.5#0.5#0.005
# 		num_iterations=1#500
# 		pgd_step_size=eps_val#/150#2.5/80#0.005/80#0.00004#0.02#eps_val/40  #0.02/80 works well

# 		orth_step_size=eps_val#/150#2.5/80#0.005/80#0.00004#0.02#eps_val/40
# 		out_iterations=num_iterations
# 		in_iterations=1
# 		norm=float(np.inf)

# 		#apgdt = APGDAttack(model, eps=0.20, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
# 		orthmll=OrthMLLAttacks()
# 		criterion=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')

# 		#for epoch in range(start_epoch,start_epoch+num_epochs):
# 		for es in range(0,1):
# 			#current_target_value=1.0-current_target_value
# 			avg_meter=AverageMeter()
# 			# if epoch in mstones:
#  			# 	lrval=lrval*0.1

# 			model.train()
# 			model.apply(self.set_bn_eval)
# 			model.model.eval()		# do not update BN
# 			#torch.set_grad_enabled(True)
# 			#self.logger.write('\nModel Training:')
# 			totals={i:torch.Tensor([1]) for i in range(num_classes)}
# 			success={i:torch.Tensor([0]) for i in range(num_classes)}
# 			flipped={i:torch.Tensor([0]) for i in range(num_classes)}
# 			perf_matches={i:torch.Tensor([0]) for i in range(num_classes)}

# 			#mytargetlabels=[7]
# 			#mytargetlabels=[1,4,8,15,17]
# 			alltargetlabels=[[i] for i in range(num_classes)] #batch size = 40 x 3
# 			#alltargetlabels=[[0,15,18],[0,10,11],[0,15,16]] #45, 26, 24, 59 batch size: 24
# 			#alltargetlabels=[[0,14,16],[0,15,18],[0,10,11],[0,15,16],[0,15,19],[0,11,12],[15,16,17],[15,17,18]] #batch size: 10 x 4

# 			#alltargetlabels=[[0,5],[0,12],[15,18],[0,9],[0,18],[0,4], [0,11],[0,8], [15,19], [0,14],[15,16],[0,10],[10,11],[0,15] ] #40 x 3
# 			#182, 58,110, 148, 96,146,90, 137, 40,69, 95, 57,51, 70    
# 			#
# 			#target_on_off=1
# 			max_batch=80
# 			max_num_batches=10
# 			#nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))
# 			allset={}
# 			#for current_target_labels in alltargetlabels:
# 			#for current_target_labels,target_on_off in itertools.product(alltargetlabels,[0,1]):
			
# 			#pert_param = torch.nn.Parameter(torch.zeros((3,224,224),dtype=torch.float32).cuda())
# 			pert_param = torch.nn.Parameter(torch.zeros((2,self.num_classes,3*224*224),dtype=torch.float32).cuda())
# 			pert_param.requires_grad=True
			
# 			train_optimizer = torch.optim.Adam([pert_param], lr=float(1e-2))#,weight_decay=1e-4)
# 			scheduler = MultiStepLR(train_optimizer, milestones=[5], gamma=0.1)

# 			train_optimizer.step()

# 			for current_target_labels in [[1]]:#alltargetlabels[2:]:
# 				target_on_off=0
# 				#print(current_target_labels,target_on_off)
				
# 				print('Starting target labels:',current_target_labels, ', Goal:',('Turn ON','Turn OFF')[target_on_off])
# 				#current_store_folder=os.path.join('/mnt/raptor/hassan/MLLFinal/stores/',str(target_on_off),'_'.join([str(i) for i in current_target_labels]))
# 				current_store_folder=os.path.join('/mnt/raptor/hassan/MLLeps/stores/','norm_'+str(norm),str(eps_val),str(target_on_off),'_'.join([str(i) for i in current_target_labels]))
# 				orth_store_folder=os.path.join(current_store_folder,'orth_l2')
# 				pgd_store_folder=os.path.join(current_store_folder,'pgd_l2')
# 				pgd_fixed_store_folder=os.path.join(current_store_folder,'pgd_fixed_l2')
# 				# aorth_store_folder=os.path.join(current_store_folder,'aorth_l2')
# 				# apgd_store_folder=os.path.join(current_store_folder,'pgd_l2')
				
# 				create_folder(os.path.join(orth_store_folder,'pert'))
# 				create_folder(os.path.join(pgd_store_folder,'pert'))
# 				create_folder(os.path.join(pgd_fixed_store_folder,'pert'))
# 				#create_folder(orth_store_folder)
# 				#create_folder(apgd_store_folder)
				
# 				for epoch in range(1000):
# 					print('Epoch:',epoch)	
# 					batch_labels=torch.empty((0,20),dtype=torch.float32).cuda()
# 					orth_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 					#aorth_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 					#apgd_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 					pgd_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 					pgd_fixed_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 					clean_outputs=torch.empty((0,20),dtype=torch.float32).cuda()

# 					batch_img_ids=[]
# 					batch_inputs=torch.empty((0,3,224,224),dtype=torch.float32).cuda()
# 					number_of_batch_iterations=0 

# 					for i, data in enumerate(tqdm(train_dataloader)):
# 						# if i==0:
# 						# 	continue
# 						# get the inputs; data is a list of [image_ids, inputs, labels]
# 						#img_ids,inputs, labels = data 
# 						train_optimizer.zero_grad()
# 						if number_of_batch_iterations>=max_num_batches:
# 							break

# 						(all_inputs,img_ids),labels=data
# 						all_inputs=all_inputs.to(device)
# 						labels=labels.to(device).float()
						
# 						model.eval()
						
# 						outputs,features = model(all_inputs,{'epsilon_norm':0.0})					
						
# 						#outputs,features = model(all_inputs,epsilon_norm=epsilon_norm,target_label=labels,target_class=torch.arange(num_classes),combine_uaps=True)
# 						labels=torch.clone(outputs).detach()
# 						labels=torch.where(labels>0,1,0).float()

# 						mytargetlabels=current_target_labels

# 						select_indices=torch.ones(size=(all_inputs.shape[0],)).type(torch.ByteTensor)
# 						for k in mytargetlabels:
# 							select_indices=torch.logical_and(select_indices,labels[:,k].cpu()==target_on_off)
# 							#select_indices=torch.logical_and(select_indices,labels[:,5].cpu()==0)


# 						select_indices=torch.where(select_indices==True)[0]
# 						#print('Total:',len(select_indices))
# 						if(len(select_indices)==0):
# 							continue 

# 						#0/0
# 						labels=labels[select_indices,:]
# 						inputs=all_inputs[select_indices,:]
# 						outputs=outputs[select_indices,:]
# 						img_ids=np.array(img_ids)[select_indices].tolist()
						
# 						batch_labels=torch.cat((batch_labels,labels),dim=0)
# 						clean_outputs=torch.cat((clean_outputs,outputs),dim=0)
# 						batch_inputs=torch.cat((batch_inputs,inputs),dim=0)

# 						if(len(inputs)==1):
# 							img_ids=[img_ids]

# 						batch_img_ids+=img_ids

# 						del inputs 
# 						if(batch_inputs.shape[0]<max_batch):
# 							continue

# 						del all_inputs 

# 						inputs=batch_inputs[:max_batch]
# 						labels=batch_labels[:max_batch]
# 						img_ids=batch_img_ids[:max_batch]
# 						clean_outputs=clean_outputs[:max_batch]

# 						#mytargetlabels=[5]
# 						jk=1
# 						to_use_selection_mask=True
# 						newlabels=labels

# 						tempsucc=0
# 						flipped_labels=0

# 						model_params={
# 						'p_norm':p_norm,
# 						'epsilon_norm':epsilon_norm,
# 						'target_label':newlabels,
# 						'combine_uaps':combine_uaps,
# 						#'target_class':(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
# 						}

# 						for ts in mytargetlabels:
# 							newlabels[:,ts]=1-newlabels[:,ts]
						
# 						#selection_mask_target=torch.zeros_like(newlabels,dtype=torch.float32)
# 						selection_mask_target=torch.zeros_like(newlabels,dtype=torch.float32)
# 						for ts in mytargetlabels:
# 							selection_mask_target[:,ts]=1.0

# 						# continue	
# 						beforestore=clean_outputs

# 						print('Target labels:',mytargetlabels)
# 						nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))
						
# 						print('\nEpsilon:',eps_val)
# 						#best_e=orthmll.autoperturb(model,inputs,newlabels,mytargetlabels,nontargetlabels,eps_val,norm,orth_step_size ,criterion, out_iterations,in_iterations)
# 						tempinputs=torch.clamp(torch.clone(inputs+pert_param).detach(),0.0,1.0)
# 						#best_e=orthmll.UMLLperturb(model,tempinputs,newlabels,mytargetlabels,nontargetlabels,eps_val,norm,orth_step_size ,criterion, out_iterations,in_iterations)
# 						#best_e=best_e.view(3,224,224)
# 						target_grad,nontarget_grad=orthmll.UMLLperturb(model,tempinputs,newlabels,mytargetlabels,nontargetlabels,eps_val,norm,orth_step_size ,criterion, out_iterations,in_iterations)
						
# 						# with torch.no_grad():
# 						# 	target_grad=torch.clone(target_grad).detach()
# 						#target_grad=target_grad.view(3,224,224)
# 						#nontarget_grad=nontarget_grad.view(3,224,224)

# 						#best_e=best_e.view(3,224,224)
# 						#print(target_grad.shape,nontarget_grad.shape,pert_param.view(-1).shape)
# 						#0/0
# 						print(target_grad.shape)
# 						#target_grad=torch.sum(target_grad.squeeze(),dim=0)
# 						pert_param.grad=-1*target_grad.view(3,224,224)
# 						#pert_param.requires_grad=True 
# 						#loss = torch.square(torch.matmul(target_grad.squeeze(),pert_param.view(-1))) #+ torch.mul()
# 						#loss=loss.sum()
# 						#loss.backward()

# 						best_e=torch.clone(pert_param.grad).detach()
# 						#print('Sum grad:',torch.sum(best_e))
# 						# print(loss.shape)(20)

# 						# loss=torch.matmul(torch.transpose(nontarget_grad,1,2),pert_param.view(-1))
# 						# print(loss.shape) 20 x 19
# 						#0/0
# 						#pert_param.grad=best_e


# 						train_optimizer.step()


# 						#normalize 
# 						with torch.no_grad():
# 							pert_param.data=normalize_vec(pert_param.data,max_norm=0.05,norm_p=norm)


# 						#best_e2=best_e.reshape(-1,150528).detach().cpu().numpy()

# 						#newinputs=torch.clamp(torch.clone(inputs).cuda()+pert_param,0.0,1.0)
# 						#newinputs=torch.clone(inputs).cuda()+pert_param
# 						newinputs=torch.clamp(torch.clone(inputs.cuda()+pert_param).detach(),0.0,1.0)
# 						#newinputs.requires_grad=True
# 						outputs,_ = model(newinputs,{'epsilon_norm':0.0})
# 						afterstore=torch.clone(outputs)
# 						koutputs=torch.where(outputs>0,1,0).float()
# 						print('Eps:',torch.min(pert_param),torch.max(pert_param))

# 						#project at the intersection of previous non-target and current non-target directions````
						
# 						#e=best_e.view(inputs.shape[0],3,224,224).cuda()
# 						e=best_e
# 						#print('Norm '+str(norm)+':',torch.linalg.vector_norm(e.view(-1),ord=norm,dim=-1))
# 						#print('Norm 1:',torch.linalg.vector_norm(torch.abs(e.view(e.shape[0],-1)),ord=1,dim=-1))
# 						success_indices=torch.where(((koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)==len(mytargetlabels))==True)[0]

# 						#print('\nSuccess:',len(success_indices),torch.min(e).item(),torch.max(e).item())

# 						#print('Success:',len(success_indices),torch.min(self.e),torch.max(self.e))
# 						scores=(koutputs==newlabels).float().sum(1)
# 						ac_scores=torch.count_nonzero(scores==koutputs.shape[1]).item()
						
# 						out=torch.abs(beforestore-afterstore)
# 						print('Orth Success: %d, All Class success: %d'%(len(success_indices),ac_scores),', min: %.3f, max: %.3f, target sum: %.3f, nontarget sum: %.3f'%(torch.min(e).item(),torch.max(e).item(),torch.sum(out[:,mytargetlabels]).item(),torch.sum(out[:,nontargetlabels]).item()))
# 						print('Scores:',scores)

						
# 						number_of_batch_iterations+=1
# 						batch_labels=torch.empty((0,20),dtype=torch.float32).cuda()
# 						orth_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
						
# 						#aorth_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 						#apgd_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 						clean_outputs=torch.empty((0,20),dtype=torch.float32).cuda()

# 						batch_img_ids=[]
# 						batch_inputs=torch.empty((0,3,224,224),dtype=torch.float32).cuda()
# 						gc.collect()
# 						torch.cuda.empty_cache()

# 					if((epoch+1)%10==0):
# 						print('Saving for epoch',epoch)
# 						torch.save(pert_param.data,os.path.join('/mnt/raptor/hassan/weights/umll/1/',str(epoch)+'.pt'))

# 					scheduler.step()

						


	
# args=parser.parse_args()
# trainer=Trainer(args.configfile,args.expname)
# trainer.train()

