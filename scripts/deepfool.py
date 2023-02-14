import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from tqdm import tqdm 

# def check_condition(output,target_labels,target_class):
#     #fs[0,label] > 0 and present_to_absent) or (fs[0,label] < 0 and not present_to_absent))
#     fs=torch.where(torch.clone(output)>0,1,0)
#     mask=fs[:,target_class]!=target_labels[:,target_class]
#     if(torch.count_nonzero(mask)>0):
#         return True, mask

#     return False, mask

# def deepfool(image, net, target_labels,target_class, num_classes=20, overshoot=0.02, max_iter=10):
#     # deepfool with batch
#     """
#        :param image: Image of size HxWx3
#        :param net: network (input: images, output: values of activation **BEFORE** softmax).
#        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
#        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
#        :param max_iter: maximum number of iterations for deepfool (default = 50)
#        :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
#     """
#     is_cuda = torch.cuda.is_available()

#     if is_cuda:
#         #print("Using GPU")
#         image = image.cuda()
#         net = net.cuda()
#     else:
#         print("Using CPU")

#     mul_factor=(2*target_labels[:,target_class])-1
#     #f_image,_ = net.forward(Variable(image[None, :, :, :], requires_grad=True),epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
#     f_image,_ = net.forward(Variable(image, requires_grad=True),epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
#     print(f_image.shape)
#     f_image=f_image.view(f_image.shape[0],-1)

#     #I = (np.array(f_image)).flatten().argsort()[::-1]
#     #I = (np.array(f_image)).flatten()
#     I = torch.clone(f_image)

#     #I = I[0:num_classes]
#     #label = I[0]
    

#     input_shape = image.shape
#     pert_image = torch.clone(image)
    
#     w = torch.zeros(input_shape)
#     r_tot = torch.zeros(input_shape).cuda()

#     loop_i = 0

#     x = pert_image
#     #x.requires_grad=True
#     x = Variable(x, requires_grad=True)
#     fs,_ = net.forward(x,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())

    
#     k_i = target_class

#     while loop_i < max_iter:
#         condition, mask = check_condition(fs,target_labels,target_class)
#         #print('Condition:',condition)
#         mask=mask.float()
#         #print(mask.T)

#         if(not condition):
#             break

#         pert = np.inf
        
#         #torch.autograd.grad(fs[:,target_class],inputs=x)

#         #print(x.grad.shape)
#         # for kp in range(fs.shape[0]):
#         #     fs[kp, target_class].backward(retain_graph=True)

#         # grad_orig = torch.autograd.grad(fs[:,target_class],x)[0].data

#         grad_orig=torch.autograd.grad(fs[:,target_class],x,grad_outputs=torch.ones_like(fs[:,target_class]),create_graph=True)[0].data

#         #print('Grad orig shape:',grad_orig.shape)
#         #print(mul_factor.shape)
#         #fs[0, label].backward()
#         #grad_orig = x.grad.data.cpu().numpy().copy()
        

#         grad_orig=grad_orig.view(grad_orig.shape[0],-1)
#         #print(grad_orig[:5,:5])
        
#         w = torch.mul(mul_factor,grad_orig)

#         f_k = torch.mul(mul_factor, fs[:, target_class]).data

#         #print('f_k:',f_k.shape)

#         # if(present_to_absent):
#         #     w = -1*grad_orig
#         #     f_k = (-1* fs[0, target_label]).data.cpu().numpy()
#         # else:
#         #     w = grad_orig
#         #     f_k = (fs[0, target_label]).data.cpu().numpy()
        
#         #normval = torch.sign(U) * torch.minimum(torch.abs(U), torch.ones_like(U)*max_norm)
#         normval=torch.linalg.vector_norm(w.view(w.shape[0],-1),ord=2,dim=1).view(-1,1)+1e-9
#         pert = abs(f_k)/normval
#         #torch.linalg.norm(w.flatten())
#         #print('Pert shape:',pert.shape)
#         #print('Pert norm:',torch.linalg.vector_norm(pert.view(w.shape[0],-1),ord=float('inf')))

#         # compute r_i and r_tot
#         # Added 1e-4 for numerical stability
#         #r_i =  torch.mul(mask,((pert+1e-4) * w / torch.linalg.vector_norm(w.view(w.shape[0],-1),ord=float('inf')))).view(-1,3,224,224)
        
#         r_i =  torch.mul(mask,((pert+1e-4) * w/normval)).view(-1,3,224,224)
#         #print('r_i shape:',r_i.shape,mask.shape)
#         r_tot = r_tot + r_i

#         if is_cuda:
#             pert_image = image + (1+overshoot)*r_tot.cuda()
#         else:
#             pert_image = image + (1+overshoot)*r_tot

#         x = Variable(pert_image, requires_grad=True)
#         fs,_ = net.forward(x,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())

#         loop_i += 1
#         if x.grad is not None:
#             x.grad.zero_()

#     r_tot = (1+overshoot)*r_tot

#     return r_tot, pert_image

# def deepfool(image, net, target_label, num_classes=20, overshoot=0.02, max_iter=10,present_to_absent=True):

#     """
#        :param image: Image of size HxWx3
#        :param net: network (input: images, output: values of activation **BEFORE** softmax).
#        :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
#        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
#        :param max_iter: maximum number of iterations for deepfool (default = 50)
#        :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
#     """
#     is_cuda = torch.cuda.is_available()

#     if is_cuda:
#         #print("Using GPU")
#         image = image.cuda()
#         net = net.cuda()
#     else:
#         print("Using CPU")


#     #f_image,_ = net.forward(Variable(image[None, :, :, :], requires_grad=True),epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
#     f_image,_ = net.forward(Variable(image[None, :, :, :], requires_grad=True),epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
#     f_image=f_image.data.cpu().numpy().flatten()

#     #I = (np.array(f_image)).flatten().argsort()[::-1]
#     I = (np.array(f_image)).flatten()

#     I = I[0:num_classes]
#     #label = I[0]
#     label = target_label

#     input_shape = image.cpu().numpy().shape
#     pert_image = copy.deepcopy(image)
#     w = np.zeros(input_shape)
#     r_tot = np.zeros(input_shape)

#     loop_i = 0

#     x = pert_image[None, :]
#     #x.requires_grad=True
#     x = Variable(x, requires_grad=True)
#     fs,_ = net.forward(x,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())

    
#     k_i = label

#     while ((fs[0,label] > 0 and present_to_absent) or (fs[0,label] < 0 and not present_to_absent)) and loop_i < max_iter:
        
#         pert = np.inf
#         fs[:, label].backward(retain_graph=True)
#         grad_orig = torch.autograd.grad(fs[0,label],x)[0].data.cpu().numpy().copy()

        
#         #fs[0, label].backward()
#         #grad_orig = x.grad.data.cpu().numpy().copy()


#         if(present_to_absent):
#             w = -1*grad_orig
#             f_k = (-1* fs[0, target_label]).data.cpu().numpy()
#         else:
#             w = grad_orig
#             f_k = (fs[0, target_label]).data.cpu().numpy()

#         pert = abs(f_k)/np.linalg.norm(w.flatten())


        

#         # compute r_i and r_tot
#         # Added 1e-4 for numerical stability
#         r_i =  (pert+1e-4) * w / np.linalg.norm(w)
#         r_tot = np.float32(r_tot + r_i)

#         if is_cuda:
#             pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
#         else:
#             pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

#         x = Variable(pert_image, requires_grad=True)
#         fs,_ = net.forward(x,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())

#         loop_i += 1
#         if x.grad is not None:
#             x.grad.zero_()

#     r_tot = (1+overshoot)*r_tot

#     return r_tot, pert_image

def check_condition(outputs,newlabels,selection_mask):

    #fs[0,label] > 0 and present_to_absent) or (fs[0,label] < 0 and not present_to_absent))
    outputs=torch.where(outputs>0,1,0)
    
    out=torch.mul((outputs==newlabels).float(),selection_mask).sum(dim=1)

    return out==selection_mask.sum(dim=1)



    fs=torch.where(torch.clone(output)>0,1,0)
    mask=fs[:,target_class]!=target_labels[:,target_class]
    if(torch.count_nonzero(mask)>0):
        return True, mask

    return False, mask

def deepfool(imgs, model, newlabels, mytargetlabels,use_selection_mask, num_classes=20, overshoot=0.02, max_iter=10):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    if use_selection_mask:
        selection_mask=torch.zeros_like(newlabels)
        for jk in mytargetlabels:
            selection_mask[:,jk]=1.0
    else:
        selection_mask=torch.ones_like(newlabels)

    #f_image,_ = net.forward(Variable(image[None, :, :, :], requires_grad=True),epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
    images=Variable(torch.clone(imgs).detach(),requires_grad=True)

    f_images,_ = model(images,{'epsilon_norm':0.0})

    I=f_images.view(f_images.shape[0],-1)
    
    #label = target_label

    input_shape = images.shape
    all_inputs=torch.clone(images)
    #pert_images = torch.clone(images)
    pert_images = torch.zeros_like(images)
    r_tot = torch.zeros_like(images)

    loop_i = 0

    #all_inputs = pert_images
    #x.requires_grad=True

    #x = Variable(all_inputs, requires_grad=True)
    x=torch.clone(all_inputs).detach()
    x.requires_grad=True
    all_outputs,_ = model(x,{'epsilon_norm':0.0})

    pert = float('inf')

    #while ((fs[0,label] > 0 and present_to_absent) or (fs[0,label] < 0 and not present_to_absent)) and loop_i < max_iter:
    all_grad_dir=torch.mul((newlabels*2)-1,selection_mask)
    
    select_indices=torch.ones((pert_images.shape[0],),dtype=torch.bool)

    for loop_i in tqdm(range(max_iter)):
        
        #mytargetgrads=[]
        #x=pert_images[select_indices,:,:,:]
        outputs=all_outputs[select_indices,:]
        grad_dir=all_grad_dir[select_indices,:]

        r_is=[]

        for idx_ts,ts in enumerate(mytargetlabels):

            gradval=torch.autograd.grad(outputs=outputs[:,ts].sum(),inputs=x,retain_graph=False)[0].detach()
            
            #print(gradval.shape)
            #print(gradval.shape,torch.sum(gradval.view(gradval.shape[0],-1),dim=-1))
            gradval=gradval[select_indices,:,:,:]
            gradval=gradval*grad_dir[:,ts][:,None,None,None]
            
            #mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))
            #mytargetgrads.append(gradval)

            w=gradval 
            f_k = outputs[:,ts]

            norms=torch.linalg.vector_norm(w.view(w.shape[0],-1),ord=2)
            pert = torch.abs(f_k)/norms
            r_i =  w*((pert+1e-4)/ norms)[:,None,None,None]
            r_is.append(r_i)
        

        r_is=torch.stack(r_is,dim=-1).sum(dim=-1).squeeze()
        
        r_tot[select_indices,:,:,:]+=r_is.detach()

        #r_tot = r_tot + r_is
        
        pert_images = all_inputs.detach() + (1+overshoot)*r_tot.cuda().detach()

        #x.grad.zero_()
        model.zero_grad()
        #x.zero_grad()
        
        x=torch.clone(pert_images).detach()
        x.requires_grad=True

        #x = Variable(pert_images, requires_grad=True)
        all_outputs,_ = model(x,{'epsilon_norm':0.0})

        select_indices=check_condition(all_outputs,newlabels,selection_mask)==False
        #print('Selection indices:',torch.count_nonzero(select_indices).cpu().item(),torch.sum(r_tot).cpu().item())
        #print(select_indices)

        #loop_i += 1
        #if x.grad is not None:
        #print(torch.linalg.vector_norm(r_tot.view(r_tot.shape[0],-1),ord=float('inf'),dim=-1).cpu().numpy())
        if torch.count_nonzero(select_indices)==0:
            break

    r_tot = (1+overshoot)*r_tot

    return r_tot, pert_images, 