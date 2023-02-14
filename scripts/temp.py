
import pandas as pd  
import numpy as np 
from scipy.linalg import subspace_angles
import torch 
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
weights=torch.from_numpy(np.load('/mnt/raptor/hassan/data/model_uniform_subspace_close.npy'))

targetclasses=[0,11]
num_classes=len(targetclasses)
Uk=[weights[1,c,:,:].squeeze().T for c in targetclasses]


Us=[u/torch.linalg.vector_norm(u,ord=2,dim=0)[None,:] for u in Uk]

print(np.rad2deg(subspace_angles(Us[0],Us[1])))
0/0
A = torch.cat([Us[0],Us[1]],dim=1)


u,s,v=torch.svd(A)
print('Singular values',s)
print(u.shape)
0/0
z=u[:,0]
print(torch.dot(z,u3.real.T))

# print(Us[0].shape)
# print(torch.matmul(Us[0].T,Us[0]))
# 0/0


# for i in Us:
# 	for j in Us:
# 		#print(i.shape,torch.linalg.vector_norm(i,ord=2,dim=0)[None,:].shape,torch.linalg.vector_norm(i,ord=2,dim=0))
# 		#print(i,torch.linalg.vector_norm(i,ord=2,dim=0)[None,:],torch.linalg.vector_norm(i,ord=2,dim=0)[None,:].shape)
# 		i=i/torch.linalg.vector_norm(i,ord=2,dim=0)[None,:]
# 		j=j/torch.linalg.vector_norm(j,ord=2,dim=0)[None,:]
# 		#print(i,torch.linalg.vector_norm(i,ord=2,dim=0)[None,:],torch.linalg.vector_norm(i,ord=2,dim=0)[None,:].shape)
# 		#0/0
# 		print(torch.dot(i[:,0],j[:,0]),torch.dot(i[:,1],j[:,1]),torch.dot(i[:,0],j[:,1]),torch.dot(i[:,1],j[:,0]))
		
# 0/0

A=torch.zeros((2*len(targetclasses),2*len(targetclasses)),dtype=torch.float32)

for i in range(num_classes):
	for j in range(i+1,num_classes):
		A[i*2:i*2+2,j*2:j*2+2]=torch.matmul(Us[i].T,Us[j])
		A[j*2:j*2+2,i*2:i*2+2]=A[i*2:i*2+2,j*2:j*2+2]

print(A)

# print(np.rad2deg(subspace_angles(Us[0],Us[1])))

# # class0=weights[0,0,:,:].squeeze().T
# # class5=weights[0,12,:,:].squeeze().T

# # A=torch.zeros((4,4),dtype=torch.float32)
# # A[:2,2:]=torch.matmul(class0.T,class5)
# # A[2:,:2]=torch.matmul(class5.T,class0)

# U,S,V=torch.svd(A)
# print(S)
# 0/0

L, EV = torch.linalg.eig(A)
print(EV.numpy())
print('EIGEN VALUES:',L)


u1=torch.sum(Us[0]*EV[0:2,0][None,:],dim=1)
u1=u1/torch.linalg.vector_norm(u1,ord=2)

u2=torch.sum(Us[1]*EV[2:,0][None,:],dim=1)
u2=u2/torch.linalg.vector_norm(u2,ord=2)

u3=u1+u2
u3=u3/torch.linalg.vector_norm(u3,ord=2)

#A = torch.stack([Us[0],Us[1]]).T
print(Us[0].shape)
A = torch.cat([Us[0],Us[1]],dim=1)



u,s,v=torch.svd(A)
print('Singular values',s)

z=u[:,0]
print(torch.dot(z,u3.real.T))

0/0
print(z.shape)



#print(Us[0].shape)


vec=torch.matmul(Us[0].T,z)
vec2=torch.matmul(Us[0],Us[0].T)#,z)
vec2=torch.matmul(vec2,z)

norm1=torch.linalg.vector_norm(vec,ord=2)
norm2=torch.linalg.vector_norm(vec2,ord=2)
#print(vec2.shape)
print(norm1,norm2)
#0/0
print('First vector:',torch.square(norm1)/norm2)



# for k in range(1,)
# vec=torch.matmul(Us[0].T,u[:,1])
# print('Second vector:',torch.linalg.vector_norm(vec,ord=2))

# vec=torch.matmul(Us[0].T,u[:,1])
# print('Third vector:',torch.linalg.vector_norm(vec,ord=2))

# vec=torch.matmul(Us[0].T,u[:,1])
# print('Fourth vector:',torch.linalg.vector_norm(vec,ord=2))

0/0



0/0



0/0

print(torch.matmul(Us[0].T,Us[0]))
out=torch.matmul(Us[0],z)
print(out)
print(out.shape)
#print(u.shape)
0/0

torch.matmul()
print(u.shape)
0/0
print(u)
print(v)
0/0
print(A.shape)
0/0






out= torch.dot(u1,u3)
print('u1 u3:',out)
out= torch.dot(u2,u3)
print('u2 u3:',out)

#print(u1.shape,u2.shape)
out= torch.dot(u1,u2)
print('u1 u2:',out)

print(out)

print(out.shape)
0/0
print(Us[0].shape,EV.shape)

0/0
#tempU=np.copy(U)
#print(U)

# for i in range(2):
# 	for j in range(4):
# 		tempval=U[i*2:i*2+2,j]
# 		print(tempval,np.linalg.norm(tempval,ord=2))
# 		tempU[i*2:i*2+2,j]=tempval/(np.linalg.norm(tempval,ord=2)+1e-9)



print(U.numpy())
print(V.numpy())

#print(S)


0/0
a=torch.from_numpy((-2*np.random.rand(3*224*224,2))+1)
b=torch.from_numpy((-2*np.random.rand(3*224*224,2))+1)
print(a.shape,b.shape)

out=torch.matmul(a,b.T)

0/0



#b=(-2*np.random.rand(100,15))+1
print(a[:5,:])
print(b[:5,:])
a=np.linalg.qr(a)[0]
b=np.linalg.qr(b)[0]

#a=np.linalg.qr(0.002*np.sign(a))[0]
#b=np.linalg.qr(0.002*np.sign(b))[0]

# print(np.matmul(a.T,a))
# print(np.matmul(b.T,b))
# print(np.matmul(a.T,b))
# 0/0
# #b=q[:,2]
# b=np.random.rand(3,)
# b=b/np.linalg.norm(b,ord=2)
# l=[]
# s=[]
# for _ in range(20):
# 	randval=np.random.rand(2)
	
# 	newvec=np.sum(a*randval,axis=1)
# 	newvec=newvec/np.linalg.norm(newvec,ord=2)
# 	l+=[np.matmul(newvec.T,b)]
# 	s+=[np.square(newvec-b).sum().tolist()]

# 	#0/0
# 	#l.append()
# print(', '.join(['{:.3f}'.format(k) for k in l]))
# print('\n')
# print(', '.join(['{:.3f}'.format(k) for k in s]))

# #print(np.matmul(a.T,b))
# #print(np.matmul(a.T,a))
# 0/0

print(np.rad2deg(subspace_angles(a,b)))

0/0


d={'0_2': 18, '0_5': 182, '0_9': 58, '0_18': 110, '0_12': 148, '0_4': 96, '0_11': 146, '15_18': 90, '0_8': 137, '15_19': 40, '0_14': 69, '0_3': 6, '2_4': 13, '15_16': 95, '0_10': 57, '10_11': 51, '0_19': 39, '14_19': 2, '0_17': 7, '0_6': 10, '0_16': 22, '2_18': 4, '3_6': 3, '15_17': 16, '0_15': 70, '0_13': 29, '3_5': 6, '18_19': 6, '0_7': 13, '11_12': 10, '0_1': 16, '8_11': 1, '7_11': 2, '1_11': 1, '16_17': 2, '10_13': 1, '4_11': 1, '4_18': 8, '9_11': 3, '4_6': 2, '14_16': 11, '2_15': 5, '17_18': 1, '2_19': 1, '4_15': 2, '17_19': 3, '2_8': 1, '11_13': 1, '8_14': 1, '2_17': 2}
d2=dict(sorted([k for k in d.items() if k[1]>100], key=lambda item: item[1],reverse=True))

print(d2)

0/0
# for (k,v) in d.items():
# 	print(k,v)
# 	0/0


data=pd.read_hdf('/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/test_labels_AT.h5',key='df').iloc[:,:-1].to_numpy()
print(data.shape)

n=data.shape[1]
s=2**np.arange(n)
x1D=data.dot(s)
Xout=(x1D[:,None]==x1D).astype(float)
print(Xout)
print(Xout.shape)
sumval=np.sum(Xout,axis=1)
sortedidx=np.argsort(sumval)[::-1]

for idx in sortedidx[:460]:
	print(idx,data[idx,:])

print(data[:10])
print('n:',data[sortedidx[460]])

data2=np.abs(data-data[sortedidx[460]])
print(data2[:10])
0/0
x1D=data2.dot(s)
sortedval=np.sort(x1D)
print(sortedval)
a=np.argmin(sortedval>0)
print(list(sortedval[a:a+50]))
print(data[a:a+50])
#print(data[sortedidx[460]])




0/0


####################################

names=['hin',
'stride',
'padding', 
'dilation',
'kernel',
'output_padding']

# def calc(arr):
# 	#Hout = (hin - 1)*stride - 2*padding + dilation * (kernel - 1) + output_padding + 1
# 	return (arr[0] - 1)*arr[1] - 2*arr[2] + arr[3] * (arr[4] - 1) + arr[5] + 1
def calc(arr):
	#Hout = (hin - 1)*stride - 2*padding + dilation * (kernel - 1) + output_padding + 1
	return (arr[0] + 2 * arr[2] - arr[3]*(arr[4]-1) - 1)/arr[1] + 1
	#return (arr[0] - 1)*arr[1] - 2*arr[2] + arr[3] * (arr[4] - 1) + arr[5] + 1


arrs=[
]

for strides in [1,2,3]:
	for padding in [0,1]:
		for kernels in [3,5,7]:		
			arrs.append([224,strides,padding,1,kernels,0])
	
#1 7 11 27 55 111 224
#211, 213, 215, 209, 
for a in arrs:
	print(', '.join([names[t]+':'+str(a[t]) for t in range(6)]),' - ', calc(a))




#hin:3, stride:1, padding:0, dilation:1, kernel:5, output_padding:0  -  7
# 7 -> 13, 15
# 13 -> 27, 29
# 15 -> 31 33 35
# 27-> 55 57 59
# 29 -> 59 61 63
# 31 -> 61 63 65 67
# 33 -> 65 67 69 71
# 35 -> 69, 71, 73, 75

# 55-> 111, 113

# 111 -> 224 hin:111, stride:2, padding:0, dilation:1, kernel:3, output_padding:1