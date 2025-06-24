from __future__ import print_function
import os
import sys
sys.path.append('./../')
#sys.path.append(os.getcwd())
from utils.utility import * 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 



class UAPLoss(nn.Module):
    #def __init__(self,loss_hyperparams,metadata,tree,common_parents_tree,common_hierarchy_tree,use_target_indices=False,isattack=False):
    def __init__(self,params):
        super(UAPLoss,self).__init__()
        #self.bceloss=nn.BCEWithLogitsLoss()
        self.bcescale=float(params['bcescale'])
        self.pwscale = float(params['pwscale'])
        self.orthscale=float(params['orthscale'])
        self.usumscale=float(params['usumscale'])
        self.upsumscale=float(params['upsumscale'])
        self.indscale=float(params['indscale'])
        self.weightscale=float(params['weightscale'])
        self.bceloss=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')
        #self.bceloss=AsymmetricLossOptimized()
    
    def compute_orthogonal_loss(self,model,target_class,use_selection_mask):
        # get UAPs

        #Outputs,F=model.get_UAPs_features()
        U=model.get_params()

        orthloss=torch.square(U-U[0,target_class,:]).sum(dim=2)
        orthloss+=torch.square(U-U[1,target_class,:]).sum(dim=2)
        print(orthloss.shape)

        with torch.no_grad():
            mask=torch.ones_like(orthloss)
            mask[:,target_class]=0.0

        orthloss=torch.mul(orthloss,mask).sum()

        orthloss=orthloss/(U.shape[0]*U.shape[1]*U.shape[2])
        
        
        return orthloss 

    def forward(self,outputs,labels,target_classes,selection_mask,model=None,getfull=False):

        bceloss=torch.mul(self.bceloss(outputs,labels),selection_mask)
        
        if not getfull:
            bceloss=bceloss.sum()
            
        bceloss=self.bcescale*(bceloss/outputs.shape[0])
        
        losses_dict={
        'bce_loss':bceloss.sum(),
        #'norm_loss':normloss
        }

        lossval=bceloss#+self.weightscale*normloss
        losses_dict['total_loss']=lossval.sum()

        return losses_dict,lossval#bceloss
        
        