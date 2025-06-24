import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from tqdm import tqdm 

# def deepfoolbatch(images, net, target_label, max_norm=0.05,num_classes=20, overshoot=0.01, max_iter=400,present_to_absent=None):

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
#         images = images.cuda()
#         net = net.cuda()
#     else:
#         print("Using CPU")


#     #f_image,_ = net.forward(Variable(image[None, :, :, :], requires_grad=True),epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
#     #f_image,_ = net.forward(Variable(image[None, :, :, :], requires_grad=True),epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
#     f_image,_ = net.forward(Variable(images, requires_grad=True))
#     print(f_image.shape)
#     f_image=f_image.data.cpu().numpy().flatten()

#     #I = (np.array(f_image)).flatten().argsort()[::-1]
#     I = (np.array(f_image)).flatten()

#     print(I.shape)
#     0/0

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
# #     fs,_ = net.forward(x,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
#     fs,_ = net.forward(x)#,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())

    
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
        
#         r_tot=normalize_vec_np(r_tot,max_norm=max_norm)
        
        

#         if is_cuda:
#             pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
#         else:
#             pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)
        
#         pert_image=torch.clip(pert_image,0.0,1.0)
#         x = Variable(pert_image, requires_grad=True)
#         fs,_ = net.forward(x)#,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())

#         loop_i += 1
#         if x.grad is not None:
#             x.grad.zero_()

#     r_tot = (1+overshoot)*r_tot
    
#     r_tot=normalize_vec_np(r_tot,max_norm=max_norm)
#     print('Loops:',loop_i)
#     return r_tot, pert_image

def normalize_vec_np(x, max_norm):
    x = np.sign(x) * np.minimum(np.abs(x), np.ones_like(x)*max_norm)
    return x


def deepfool(image, net, target_label, max_norm=0.05,num_classes=20, overshoot=0.01, max_iter=400,present_to_absent=True):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        #print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    #f_image,_ = net.forward(Variable(image[None, :, :, :], requires_grad=True),epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
    #f_image,_ = net.forward(Variable(image[None, :, :, :], requires_grad=True),epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
    f_image,_ = net.forward(Variable(image[None, :, :, :], requires_grad=True))
    f_image=f_image.data.cpu().numpy().flatten()
    # print(f_image)
    origpred=f_image[target_label]
    # print('original prediction:',origpred)

    origpred=np.where(origpred>0,1,0)

    #I = (np.array(f_image)).flatten().argsort()[::-1]
    I = (np.array(f_image)).flatten()

    I = I[0:num_classes]
    #label = I[0]
    label = target_label

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = pert_image[None, :]
    #x.requires_grad=True
    x = Variable(x, requires_grad=True)
#     fs,_ = net.forward(x,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())
    fs,_ = net.forward(x)#,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())

    
    k_i = label

    while ((fs[0,label] > 0 and present_to_absent) or (fs[0,label] < 0 and not present_to_absent)) and loop_i < max_iter:
        
        pert = np.inf
        fs[:, label].backward(retain_graph=True)
        grad_orig = torch.autograd.grad(fs[0,label],x)[0].data.cpu().numpy().copy()

        
        #fs[0, label].backward()
        #grad_orig = x.grad.data.cpu().numpy().copy()


        if(present_to_absent):
            w = -1*grad_orig
            f_k = (-1* fs[0, target_label]).data.cpu().numpy()
        else:
            w = grad_orig
            f_k = (fs[0, target_label]).data.cpu().numpy()

        pert = abs(f_k)/np.linalg.norm(w.flatten())

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        
        r_tot=normalize_vec_np(r_tot,max_norm=max_norm)
        
        

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)
        
        pert_image=torch.clip(pert_image,0.0,1.0)
        x = Variable(pert_image, requires_grad=True)
        fs,_ = net.forward(x)#,epsilon_norm=0.0,target_label=torch.zeros((1,num_classes),dtype=torch.float32),target_class=torch.Tensor([0]).long())

        loop_i += 1
        if x.grad is not None:
            x.grad.zero_()

    
    finalpred=fs[0,target_label].detach().cpu().numpy()
    # print('Final pred:',finalpred)
    finalpred=np.where(finalpred>0,1,0)

    r_tot = (1+overshoot)*r_tot
    
    r_tot=normalize_vec_np(r_tot,max_norm=max_norm)
    print('Loops:',loop_i,', Res:',finalpred==(1-origpred))
    return r_tot, pert_image, (finalpred==(1-origpred)).tolist()