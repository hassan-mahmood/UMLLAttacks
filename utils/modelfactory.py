
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
# from MLGCN.models import *
sys.path.append('ASL/')
#from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from src.models import create_model
from scipy.linalg import subspace_angles
pickle.HIGHEST_PROTOCOL = 5
torch.set_printoptions(edgeitems=27)
np.set_printoptions(precision=3)

seed=999
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def createmodel(model_name,model_path,logger,dataset_name,args):
	
	if dataset_name=='OpenImages':
		if model_name=='asl':
			tempargs={'image_size':args.image_size,'model_name':model_name,
			'sep_features':0,
			'use_ml_decoder':False,#(False,True)[model_name=='tresnet_mldecoder'],
			'num_of_groups':-1,
			'decoder_embedding': 768,
			'load_head':True,
			'do_bottleneck_head':True,
			'zsl':0,
			'model_path':model_path,
			'num_classes':args.num_classes,'workers':0
			}
			tempargs=argparse.Namespace(**tempargs)
			state = torch.load(model_path, map_location='cpu')
			# tempargs.do_bottleneck_head = True
			model = create_model(tempargs).cuda()
			logger.write('\nLoading main model from:',model_path)
			model.load_state_dict(state['model'], strict=True)
			#model.load_state_dict(state['model_state'], strict=True)
			# CUDA_VISIBLE_DEVICES=0 python nusscripts/oracle2.py --configfile configs/nus_mldecoder.ini mldecoder_oracle2_fixed_target2
			logger.write('\n loaded ASL model:',model_path)
			# print('ASL Model loaded')

	elif model_name=='asl':
		tempargs={'image_size':args.image_size,'model_name':model_name,
		'sep_features':0,
		'use_ml_decoder':False,#(False,True)[model_name=='tresnet_mldecoder'],
		'num_of_groups':-1,
		'decoder_embedding': 768,
		'load_head':False,
		'do_bottleneck_head':False,
		'zsl':0,
		'model_path':model_path,
		'num_classes':args.num_classes,'workers':0
		}
		tempargs=argparse.Namespace(**tempargs)
		state = torch.load(model_path, map_location='cpu')
		# tempargs.do_bottleneck_head = True
		model = create_model(tempargs).cuda()
		logger.write('\nLoading main model from:',model_path)
		model.load_state_dict(state['model'], strict=True)
		#model.load_state_dict(state['model_state'], strict=True)
		# CUDA_VISIBLE_DEVICES=0 python nusscripts/oracle2.py --configfile configs/nus_mldecoder.ini mldecoder_oracle2_fixed_target2
		logger.write('\n loaded ASL model:',model_path)
		# print('ASL Model loaded')

	elif model_name=='mldecoder':

		# for MSCOCO
		tempargs={'image_size':args.image_size,'model_name':model_name,
		'sep_features':0,
		'use_ml_decoder':True,#(False,True)[model_name=='mldecoder'],
		'num_of_groups':-1,
		'decoder_embedding': 768,
		'load_head':True,
		'do_bottleneck_head':True,
		'zsl':0,
		'model_path':model_path,
		'num_classes':args.num_classes,'workers':0
		}
		tempargs=argparse.Namespace(**tempargs)
		state = torch.load(model_path, map_location='cpu')
		# tempargs.do_bottleneck_head = True
		model = create_model(tempargs).cuda()
		logger.write('\nLoading main model from:',model_path)
		if dataset_name=='NUSWIDE':
			model.load_state_dict(state['model_state'], strict=True)
		else:
			model.load_state_dict(state['model'], strict=True)
		# print('MLDecoder Model has been loaded')
		logger.write('\n loaded MLDecoder model:',model_path)
		
	else:
		print('Model name not specified.')
		0/0
	return model

