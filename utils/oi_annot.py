import torch
import numpy as np 
from tqdm import tqdm
import pandas as pd
import os 
import pickle

state = torch.load('/mnt/raptor/hassan/ModelWeights/OpenImages/Open_ImagesV6_TRresNet_L_448.pth', map_location='cpu')
idx_to_classname=state['idx_to_class']
# # print([k.strip('\"\'') for k in idx_to_classname.values()])
# pickle.dump({[k.strip('\"\'') for k in idx_to_classname.values()]}, open('/mnt/raptor/hassan/datasets/OpenImages/metadata/classnames','wb'))
# 0/0
classnames_to_idx={str(v.strip("\'\"\'")).lower():k for k,v in idx_to_classname.items()}
#classnames_to_idx={str(v):k for k,v in idx_to_classname.items()}

mainimgdirpath='/mnt/raptor/hassan/datasets/OpenImages/Images/'
mode='test'
checkmode='val'
midmapping='/mnt/raptor/hassan/datasets/OpenImages/metadata/oidv6-class-descriptions.csv'
mid_to_classnames_data=pd.read_csv(midmapping)
tempdict=dict(mid_to_classnames_data.values)
print(len(list(tempdict.keys())))
#print(mid_to_classnames)
mid_to_classnames={k:str(v.strip("\'\"\'")).lower() for k,v in tempdict.items()}
print(len(list(mid_to_classnames.keys())))

classnamekeys=[l.lower() for l in mid_to_classnames.values()]
midkeys=[l.lower() for l in mid_to_classnames.keys()]

#i need mid to classid

#idx_to_mid={k:classnames_to_mid[v.strip('\"')] for k,v in idx_to_classname.items()}
mid_to_idx={}

#classnamekeys=

for idx_k,v in idx_to_classname.items():
    v=v.strip('\"\'').lower()
    
    if v not in classnamekeys:
        # print(v,'not in classnames')
        for idxclassname,k in enumerate(classnamekeys):
            if v in k:
                print(v,':',k,classnamekeys[idxclassname],idx_k,idxclassname)
                mid_to_idx[midkeys[idxclassname]]=idx_k
    else:
        idxclassname=classnamekeys.index(v)
        mid_to_idx[midkeys[idxclassname]]=idx_k
        #idx_to_mid[idx_k]=classnames_to_mid[v]
#idx_to_mid={k:classnames_to_mid[)] }


#data[5649752:]
from ast import literal_eval
# start_idx=0
# end_idx=5649751 # till train index
start_idx=5649753
end_idx=5764393
imgids=[]
#testlabels=np.zeros(shape=(end_idx-start_idx,9605),dtype=np.float32)
testlabels=[]
# allindices=[]
count=0
midkeys=list(mid_to_idx.keys())
filepath='/mnt/raptor/hassan/datasets/OpenImages/labels/data.csv'
data=pd.read_csv(filepath)
mainimgdir=''
countfile=0
for kidx,k in enumerate(tqdm(range(start_idx,end_idx))):
    if((kidx+1)%50000==0):
        print('Completed:',kidx,', count:',count)
        np.save('/mnt/raptor/hassan/datasets/OpenImages/labels/'+mode+'files/'+mode+'labels'+str(countfile)+'.npy',np.array(testlabels).squeeze())
        image_paths=[os.path.join(mainimgdirpath,mode,s) for s in imgids]
        pickle.dump(image_paths,open('/mnt/raptor/hassan/datasets/OpenImages/labels/'+str(mode)+'imgids'+str(countfile),'wb'))

        # newdata=pd.DataFrame(np.array(testlabels).squeeze())
        # newdata['imgids']=imgids
        # newdata.to_hdf('/mnt/raptor/datasets/OpenImages/labels/trainfiles/trainlabels'+str(countfile)+'.h5',key='df')
        countfile+=1
        testlabels=[]
        imgids=[]
        
    if(os.path.exists(os.path.join(mainimgdirpath,data.iloc[k,0]))):
        assert(data.iloc[k,-1]==checkmode)
        imgids.append(os.path.basename(data.iloc[k,0]))
        count+=1
        labels=literal_eval(data.iloc[k,1])
        labelindices=[mid_to_idx[l] for l in labels if l in midkeys]
        # allindices+=labelindices
        templabels=np.zeros(shape=(1,9605),dtype=np.float32)
        templabels[0,labelindices]=1.0
        
        neglabels=literal_eval(data.iloc[k,2])
        labelindices=[mid_to_idx[l] for l in neglabels if l in midkeys]
        # allindices+=labelindices
        templabels[0,labelindices]=-1.0
        
        testlabels.append(templabels)
        
    else:
        pass

print('Completed:',kidx,', count:',count)
# newdata=pd.DataFrame(np.array(testlabels).squeeze())
# newdata['imgids']=imgids
# newdata.to_hdf('/mnt/raptor/datasets/OpenImages/labels/trainfiles/trainlabels'+str(countfile)+'.h5',key='df')
np.save('/mnt/raptor/hassan/datasets/OpenImages/labels/'+mode+'files/'+mode+'labels'+str(countfile)+'.npy',np.array(testlabels).squeeze())
image_paths=[os.path.join(mainimgdirpath,mode,s) for s in imgids]
pickle.dump(image_paths,open('/mnt/raptor/hassan/datasets/OpenImages/labels/'+str(mode)+'imgids'+str(countfile),'wb'))
countfile+=1
print('Total images:',count)