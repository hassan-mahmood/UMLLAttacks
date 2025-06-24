import sys 
sys.path.append('./../')
import configparser 
from utils.utility import * 
from Datasets.Dataset import *
import re 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Logger.Logger import * 
from Models.Model import *
from Losses.Losses import * 
from torch.optim.lr_scheduler import MultiStepLR


torch.backends.cudnn.deterministic = True
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)


# class DataParser:
# 	def __init__(self,params):

# 		self.experiment_name=params['experiment_name']
# 		self.mode=params['mode']
# 		configfile=params['configfile']
# 		self.dataset_name=params['dataset_name']
# 		self.parse(configfile)
		

# 	def get_section(self,config_data,section_name):
# 		tempdata=config_data._sections[section_name]
# 		return {k:tempdata[k] for k in tempdata.keys()}

# 	def parse(self,configfile):
# 		all_config_data=configparser.ConfigParser()
# 		all_config_data.read(configfile)
# 		#exp_data=all_config_data[self.experiment_name]
# 		self.experiment_data=self.get_section(all_config_data,self.experiment_name)
# 		self.loss_hyperparams = self.get_section(all_config_data,self.experiment_name+'.loss_hyperparams')
# 		self.meta_data=self.get_section(all_config_data,'metadata')
# 		self.globalvars=self.get_section(all_config_data,'globalvars')
		
# 		#self.initialize()


# 	def build(self):
# 		checkpoint_load_path=self.experiment_data['checkpoint_load_path']
# 		stats_folder=self.globalvars['stats_folder']
# 		stats_file=os.path.join(stats_folder,self.experiment_name+'_'+self.mode+'.txt')
# 		training_device=torch.device(self.globalvars['training_device'])
# 		class_names=get_pickle_data(self.meta_data['classnames_path'])
# 		writer=SummaryWriter(os.path.join(self.globalvars['log_folder'],self.experiment_name))
# 		weights_dir=os.path.join(self.globalvars['weights_dir'],self.experiment_name)
		

# 		create_folder(weights_dir)
# 		create_folder(stats_folder)

# 		num_classes=len(class_names)
# 		start_epoch=0
# 		min_val_loss=1e10
# 		logger=Logger(stats_file)
		
# 		self.allmodels={'resnet':ResNetModel}
# 		self.globalvars['num_classes']=num_classes
		
# 		model_params={
# 		'num_classes':num_classes,
# 		'mode':self.mode
# 		}
# 		all_params={}

# 		# if(combine_uaps):
# 		# 	model=UAPModelCombine(model_params)
# 		# else:
# 		# 	model=UAPModel(model_params)
# 		model=self.allmodels[self.experiment_data['modelname']](model_params)

# 		#model=GUAPModel(model_params)
# 		#optimizer = optim.Adam(model.parameters(), lr=float(self.experiment_data['lr']))#,weight_decay=1e-4)
# 		#optimizer=None
# 		optimizer = optim.SGD(model.parameters(), lr=float(self.experiment_data['lr']))#,weight_decay=1e-4)

# 		if(len(checkpoint_load_path)!=0 and os.path.exists(checkpoint_load_path)):
# 		  start_epoch,min_val_loss=restore_checkpoint(checkpoint_load_path,model,optimizer,logger)
# 		  # self.start_epoch,self.min_val_loss=restore_checkpoint(self.checkpoint_load_path,model,None,self.logger)
# 		  logger.write('Model loaded from',checkpoint_load_path)

# 		#optimizer = optim.SGD(model.parameters(), lr=float(self.experiment_data['lr']))#,weight_decay=1e-4)
# 		optimizer = optim.Adam(model.parameters(), lr=float(self.experiment_data['lr']))#,weight_decay=1e-4)

		
# 		all_params={
# 		'model':model,
# 		'optimizer':optimizer,
# 		'logger':logger,
# 		'writer':writer,
# 		'weights_dir':weights_dir,
# 		'num_classes':num_classes,
# 		'min_val_loss':float(min_val_loss),
# 		'start_epoch':int(start_epoch),
# 		'num_epochs':int(self.globalvars['num_epochs']),
# 		'device':training_device,
# 		'eps_norm':float(self.experiment_data['eps_norm']),
# 		'p_norm':float(self.experiment_data['p_norm']),
# 		'meta_data':self.meta_data,
# 		'globalvars':self.globalvars
# 		}
		
# 		if(self.mode=='train'):
# 			all_params=self.build_train(all_params)
# 		else:
# 			all_params=self.build_test(all_params)

# 		return all_params

# 	def build_train(self,all_params):
		
# 		if self.dataset_name!='oi':

# 			train_dataset=ImageDataset(self.globalvars,'train')
# 			#train_dataset2=ImageDataset(self.globalvars,'train')
# 			train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=int(self.globalvars['batch_size']), shuffle=True, num_workers=4)

# 			val_dataset=ImageDataset(self.globalvars,'val')
# 			val_dataloader=torch.utils.data.DataLoader(val_dataset, batch_size=int(self.globalvars['batch_size']), shuffle=False, num_workers=4)
		
# 		else:
# 			train_dataset=None 
# 			train_dataloader=None 
# 			val_dataset=None 
# 			val_dataloader=None

# 		lr_steps=[int(k) for k in ast.literal_eval(self.experiment_data['lr_steps'])]

# 		# Detect if we have a GPU available

# 		scheduler = MultiStepLR(all_params['optimizer'], milestones=lr_steps, gamma=0.1)
# 		criterion=nn.BCEWithLogitsLoss(reduction='none')
# 		#criterion=UAPLoss(self.loss_hyperparams)
# 		#criterion=GUAPLoss(self.loss_hyperparams)
		
# 		all_params=Merge_dict(all_params,{
# 		'scheduler':scheduler,
# 		'criterion':criterion,
# 		'train_dataset':train_dataset,
# 		#'train_dataset2':train_dataset2,
# 		'train_dataloader':train_dataloader,
# 		'val_dataset':val_dataset,
# 		'val_dataloader':val_dataloader,
# 		'batch_size':self.globalvars['batch_size'],
# 		'weight_store_every_epochs':int(self.globalvars['weight_store_every_epochs']),

# 		})

# 		return all_params


# 	def build_test(self,all_params):
		
# 		if self.dataset_name!='oi':
# 			test_dataset=ImageDataset(self.globalvars,'test')
# 			test_dataloader=torch.utils.data.DataLoader(test_dataset, batch_size=int(self.globalvars['batch_size']), shuffle=False, num_workers=0)
# 		else:

# 			test_dataset=None 
# 			test_dataloader=None 
			
# 		criterion=nn.BCEWithLogitsLoss(reduction='none')
# 		#criterion=GUAPLoss(self.loss_hyperparams)
		
# 		all_params=Merge_dict(all_params,{
# 		'criterion':criterion,
# 		'test_dataset':test_dataset,
# 		'test_dataloader':test_dataloader,
		
# 		})
# 		return all_params
		



class DataParser:
	def __init__(self,params):

		self.experiment_name=params['experiment_name']
		self.configfile=params['configfile']
		self.mode=params['mode']
		self.parse()
		

	def get_section(self,config_data,section_name):
		tempdata=config_data._sections[section_name]
		return {k:tempdata[k] for k in tempdata.keys()}

	def parse(self):
		all_config_data=configparser.ConfigParser()
		all_config_data.read(self.configfile)
		#exp_data=all_config_data[self.experiment_name]
		self.experiment_data=self.get_section(all_config_data,self.experiment_name)
		self.meta_data=self.get_section(all_config_data,'metadata')
		self.globalvars=self.get_section(all_config_data,'globalvars')
		
		
		#self.initialize()

	def build(self):
		all_params={**self.experiment_data,**self.meta_data,**self.globalvars}

		class_names=get_pickle_data(self.meta_data['classnames_path'])
		stats_folder=self.globalvars['stats_folder']
		stats_file=os.path.join(stats_folder,self.experiment_name+'_'+self.mode+'.txt')
		writer=SummaryWriter(os.path.join(self.globalvars['log_folder'],self.experiment_name))
		weights_dir=os.path.join(self.globalvars['weights_dir'],self.experiment_name)
		logger=Logger(stats_file)

		create_dir(weights_dir)
		create_dir(stats_folder)

		
		all_params['logger']=logger
		all_params['writer']=writer
		all_params['num_classes']=len(class_names)
		all_params['weights_dir']=weights_dir

		return all_params


