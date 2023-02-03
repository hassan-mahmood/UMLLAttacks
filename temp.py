import torch 
from Models.Model import * 
  

#computing svd
import os 
mainpath='/home/hassan/hassan/UAPs/grads/c2/0'
files=os.listdir(mainpath)[:1000]
allgrads=[]
for f in files:
    g=np.load(os.path.join(mainpath,f))
    
    allgrads.append(g/np.linalg.norm(g,ord=2))
    
allgrads=np.array(allgrads).T

    
u, s, vh = np.linalg.svd(allgrads, full_matrices=False)
print(list(s))

0/0

checkpoint1=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/zeromodel.pt')
checkpoint1=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/br/model-4.pt')

model_params={
'num_classes':20
}

model1=UAPModel(model_params)
model1.load_state_dict(checkpoint1['model_state'])

checkpoint2=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/br/model-8.pt')
model2=UAPModel(model_params)
model2.load_state_dict(checkpoint2['model_state'])

a=torch.rand(5,3,224,224)


# print(torch.sum(model1.U),torch.sum(model2.U))
# print(torch.linalg.vector_norm(model1.U,ord=float('inf'),dim=2))
# print(torch.linalg.vector_norm(model2.U,ord=float('inf'),dim=2))
# 0/0
# #out1,_=model1(a,epsilon_norm=0.0,target_label=None,target_class=torch.Tensor([0]).long())
# #out2,_=model2(a,epsilon_norm=0.0,target_label=None,target_class=torch.Tensor([0]).long())
# for p1,p2 in zip(model1.model.state_dict(),model2.model.state_dict()):
# 	if 'running' in p1:
# 		a=torch.sum(model1.model.state_dict()[p1])
# 		b=torch.sum(model2.model.state_dict()[p2])
# 		if(a!=b):
# 			print('Not equal',p1,p2,a,b)
# 			0/0
# 		print(a,b)
# 0/0
print(model1.U)
print(model2.U)
0/0
for p1,p2 in zip(model1.model.parameters(),model2.model.parameters()):
	a=torch.sum(p1)
	b=torch.sum(p2)
	print(a,b)
	if(a!=b):
		print('Not equal')
		break


0/0
for p in model1.named_parameters():
	print(p)
	
