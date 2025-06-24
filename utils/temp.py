#compute cooccurrence matrix

import pickle
import numpy as np
import pandas as pd
#data=pd.read_hdf('/mnt/raptor/hassan/data/nus/Labels/origtrainlabels.h5',key='df')
data=pd.read_hdf('/mnt/raptor/hassan/UAPs/orig_preds/nus/train_preds.h5',key='df')

# print(sum(pickle.load(open('/mnt/raptor/hassan/UAPs/stores/nus/target_classes/1','rb'))['target_classes']),[])
# 0/0

print('data shape:',data.shape)
arr=data.iloc[:,:-1].to_numpy()
pickle.dump(arr,open('trainpreds.npy','wb'))
0/0
# maximal_cliques = find_maximal_cliques(arr)
# print("Maximal Cliques:")
# for clique in maximal_cliques:
#     print(clique)


cooccur_matrix = np.dot(np.transpose(arr), arr)
print(cooccur_matrix)