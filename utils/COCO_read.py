import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
import pandas as pd
from tqdm import tqdm 
# from util import *

# urls = {'train_img':'http://images.cocodataset.org/zips/train2014.zip',
#         'val_img' : 'http://images.cocodataset.org/zips/val2014.zip',
#         'annotations':'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'}

def process_coco2014(annotationsfile_path,outputcategoryfile,outputlabelfile,outputimgpathfile,imagedir):
    
    # annotations_data=annotationsfile_path
    img_id = {}
    annotations_id = {}
    
    #annotations_file = json.load(open(os.path.join(annotations_data, 'instances_{}2014.json'.format(phase))))
    annotations_file=json.load(open(annotationsfile_path))
    annotations = annotations_file['annotations']
    category = annotations_file['categories']
    
    category_id = {}
    for cat in category:
        category_id[cat['id']] = cat['name']

    # cat2idx = categoty_to_idx(sorted(category_id.values()))
    cat2idx = categoty_to_idx(category_id.values())
    
    images = annotations_file['images']
    for annotation in annotations:
        if annotation['image_id'] not in annotations_id:
            annotations_id[annotation['image_id']] = set()
        annotations_id[annotation['image_id']].add(cat2idx[category_id[annotation['category_id']]])

    imagepaths=[]
    imagelabels=[]
    for img in tqdm(images):
        if img['id'] not in annotations_id:
            continue
        if not os.path.exists(os.path.join(imagedir,img['file_name'])):
            continue
        if img['id'] not in img_id:
            img_id[img['id']] = {}

        img_id[img['id']]['file_name'] = img['file_name']
        img_id[img['id']]['labels'] = list(annotations_id[img['id']])
        imagepaths.append(os.path.join(imagedir,img['file_name']))
        templabel=np.zeros(shape=(1,len(cat2idx)))

        for c in list(annotations_id[img['id']]):
            templabel[0,c]=1
        imagelabels.append(np.copy(templabel))
        
    imagelabels=np.concatenate(imagelabels)
    
    os.makedirs(os.path.dirname(outputimgpathfile),exist_ok=True)
    os.makedirs(os.path.dirname(outputlabelfile),exist_ok=True)
    os.makedirs(os.path.dirname(outputcategoryfile),exist_ok=True)

    np.save(outputlabelfile,imagelabels)
    pickle.dump(imagepaths,open(outputimgpathfile,'wb'))
    pickle.dump(cat2idx,open(outputcategoryfile,'wb'))

    
    

def categoty_to_idx(category):
    cat2idx = {}
    for cat in category:
        cat2idx[cat] = len(cat2idx)
    return cat2idx

# imagedir='/mnt/raptor/hassan/datasets/COCO/val2014/'
# inputjsonfile='/mnt/raptor/hassan/datasets/COCO/annotations/instances_val2014.json'
# outputcategoryfile='/mnt/raptor/hassan/datasets/COCO/metadata/classnames'
# outputlabelfile='/mnt/raptor/hassan/datasets/COCO/labels/val2014_labels.npy'
# outputimgpathfile='/mnt/raptor/hassan/datasets/COCO/labels/val2014_imgids'

imagedir='/mnt/raptor/hassan/datasets/COCO/train2014/'
inputjsonfile='/mnt/raptor/hassan/datasets/COCO/annotations/instances_train2014.json'
outputcategoryfile='/mnt/raptor/hassan/datasets/COCO/metadata/classnames'
outputlabelfile='/mnt/raptor/hassan/datasets/COCO/labels/train2014_labels.npy'
outputimgpathfile='/mnt/raptor/hassan/datasets/COCO/labels/train2014_imgids'

process_coco2014(inputjsonfile,outputcategoryfile,outputlabelfile,outputimgpathfile, imagedir)





