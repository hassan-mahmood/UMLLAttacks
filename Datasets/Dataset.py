import os
import pickle
pickle.HIGHEST_PROTOCOL = 3

import cv2 
import pandas as pd
#from torchvision.io import read_image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from skimage import io
from PIL import Image
import numpy as np 
from tqdm import tqdm 
import torch
from utils.utility import *
import ast 
from torchvision import transforms
from scipy.special import expit, logit
#from tree_loss import * 
torch.backends.cudnn.deterministic = True
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)




class SelectiveDataset(Dataset):
    def __init__(self, params):
        # Pass labels, image ids, image directory

        self.input_size=int(params['input_size'])
        self.images_dir=params['images_dir']
        self.all_img_ids=params['all_img_ids']
        self.all_labels=params['all_labels']
        
        self.all_image_paths=[os.path.join(self.images_dir,imgid) for imgid in self.all_img_ids]

        self.transform= transforms.Compose([
            #transforms.RandomResizedCrop(self.input_size),
            #transforms.RandomHorizontalFlip(),
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            #transforms.Normalize(meanstds['means'], meanstds['stds'])
        ])
    
    def __len__(self):
        return len(self.all_img_ids)
        #return len(self.selected_images)

    def __getitem__(self, idx):

        # img_path=os.path.join(self.images_dir,self.all_img_ids[idx])

        image = Image.open(self.all_image_paths[idx]).convert('RGB')
        label = np.array(self.all_labels[idx,:],dtype=np.float32)
        return (self.transform(image),self.all_img_ids[idx]),torch.from_numpy(label)

        #return self.get_function(idx)


class CombinedDataset(Dataset):
  def __init__(self,data_sets):
    self.datasets=data_sets
  def __getitem__(self,i):
    return tuple(d[i] for d in self.datasets)
  def __len__(self):
    return min(len(d) for d in self.datasets)


class ImageDataset(Dataset):
    def __init__(self, params):

        # self.input_size=int(params['input_size'])
        self.input_size=int(params['image_size'])
        self.num_classes=params['num_classes']
        
        dataset_name=params['dataset_name']

        self.pre_process_dataset(params)
        

        
    def pre_process_dataset(self,params):

        
        images_dir=ast.literal_eval(params['images_dir'])
        
        #self.all_images=images_dirs[0]+self.img_labels.iloc[:,-1].to_numpy()
        self.all_img_ids=pickle.load(open(ast.literal_eval(params['img_ids_file']),'rb'))
        self.all_images=[os.path.join(images_dir,s) for s in self.all_img_ids]
        self.all_labels=np.load(ast.literal_eval(params['labels_file']))
        print('Data size before:',len(self.all_img_ids),self.all_labels.shape)
        present_images=np.ones_like(self.all_labels[:,0])
        count=0
        # for imgidx,img in enumerate(self.all_img_ids):
        #     #if not os.path.exists(self.all_images[imgidx]):
        #     if os.path.getsize(self.all_images[imgidx])==0:
        #         count+=1
        #         present_images[imgidx]=0
        # print('Count:',count)
        
        # self.all_labels=self.all_labels[present_images==1,:]
        print(self.all_labels.shape)
        self.all_img_ids=list(np.array(self.all_img_ids)[present_images==1])
        self.all_images=list(np.array(self.all_images)[present_images==1])
        print('Data size after:',len(self.all_img_ids),self.all_labels.shape)

        assert(self.all_labels.shape[0]==len(self.all_img_ids))

        self.transform = transforms.Compose([
        #transforms.RandomResizedCrop(self.input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.Resize(self.input_size),
        transforms.CenterCrop(self.input_size),
        transforms.ToTensor(),
        # transforms.Normalize([0,0,0],[1,1,1])
        #transfors.Normalize(meanstds['means'], meanstds['stds'])
        ])

        
        
    def __len__(self):
        return len(self.all_images)
        #return len(self.selected_images)

    def __getitem__(self, idx):

        img_path=self.all_images[idx]
        image = Image.open(img_path).convert('RGB')
        label = np.array(self.all_labels[idx,:],dtype=np.float32)
        return (self.transform(image),self.all_img_ids[idx]),torch.from_numpy(label)

        #return self.get_function(idx)

