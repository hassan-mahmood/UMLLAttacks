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

	def test(self):
		params=self.parseddata.build()
		self.logger=params['logger']
		model=params['model']
		criterion=params['criterion']
		num_classes=params['num_classes']
		combine_uaps=params['combine_uaps']
		
		device=params['device']
		test_dataloader=params['test_dataloader']
		
		
		start_epoch=params['start_epoch']
		num_epochs=params['num_epochs']
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
		
		model = ResNetModel({'num_classes':20})
		#checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/res/model-184.pt')
		checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/res_mean/model-196.pt')
		model.load_state_dict(checkpoint['model_state'])

		model=model.cuda()
		model.eval()

		#torch.set_grad_enabled(False)
		self.logger.write('Model Evaluation:')
		vallabels=np.empty(shape=(0,num_classes),dtype=np.float32)
		valpreds=np.empty(shape=(0,num_classes),dtype=np.float32)
		dataidx=0
		#current_target_value=0.0
		with torch.no_grad():
			for i,data in enumerate(tqdm(test_dataloader)):
				#img_ids,inputs, labels = data 
				(all_inputs,img_ids),labels=data
				all_inputs=all_inputs.to(device)
				labels=labels.to(device).float()
				
				outputs,features = model(all_inputs)
				
				vallabels=np.concatenate((vallabels,labels.cpu()),axis=0)
				valpreds=np.concatenate((valpreds,outputs.detach().cpu().float()),axis=0)

				dataidx=dataidx+all_inputs.shape[0]

		now = datetime.now()
		
		#val_acc=eval_performance(valpreds,vallabels,all_class_names)
		statout=now.strftime("%d/%m/%Y %H:%M:%S")
		
		valpreds=np.where(valpreds>0,1,0)
		val_acc=accuracy_score(vallabels,valpreds)
		statout=statout+' - Test Acc: %.3f'%(val_acc)
		print(statout)
		
		



	
args=parser.parse_args()
Tester=Tester(args.configfile,args.expname)
Tester.test()

