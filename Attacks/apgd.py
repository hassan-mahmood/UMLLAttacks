
   
# Copyright (c) 2020-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
#

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

from autoattack.other_utils import L0_norm, L1_norm, L2_norm
from autoattack.checks import check_zero_gradients


def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball
    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    #u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()
    
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)
    
    s1 = -u.sum(dim=1)
    
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
    if c2.nelement != 0:
    
      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)
      
      #print(c2.shape, lb.shape)
      
      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
          
      while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2.)
        counter2 = counter4.type(torch.LongTensor)
        
        c8 = s[c2, counter2] + c[c2] < 0
        ind3 = c8.nonzero().squeeze(1)
        ind32 = (~c8).nonzero().squeeze(1)
        #print(ind3.shape)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]
        
        #print(lb, ub)
        counter += 1
        
      lb2 = lb.long()
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
    return (sigma * d).view(x2.shape)





class APGDAttack():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690
    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=1e-1,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger
    
    def init_hyperparam(self, x):
        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()
        
        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)
    
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        #loss_steps, j(current_iteration), k(self.n_iter2), loss_best
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
    
    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    #
    
    def attack_single_run(self, x, y, mycriterion, use_target_indices,selection_mask, x_init=None):
        
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)


        x_adv=x
        # if self.norm == 'Linf':
        #     t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
        #     x_adv = x + self.eps * torch.ones_like(x
        #         ).detach() * self.normalize(t)

        
        
        if not x_init is None:
            0/0
            x_adv = x_init.clone()
            
        # would be interesting but out of the focus. Thank you for bringing this to attention. we are oigng to add this to paper. 
        # Please notice that we implemented PGD known to be stronger attack[cite papers] as shown in literature. MIM is related to 
        # PGD. Relative performance is similar. FGSM we observe 
        # Thank you and we will add in explanation. 
        # Explanation
        # may be more like a suggestion. 
        # region proposed
        # we are focussing on these issues. why other context-aware are not but challenges are context-aware in our seeting is this
        # this would be a subject of different paper. We could not address in a single paper. Why not applicable?
        # uyes, we can leverage we thought about it and have ideas about it. How we integrate our method?
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv,None, None)[0]
                loss_indiv = -1*mycriterion(logits, y.type(torch.cuda.FloatTensor))
                #loss_indiv=torch.mul(loss_indiv,selection_mask).sum(1)

                loss_indiv = torch.sum(loss_indiv,1)
                #loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

                #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                #hassan * -1
                #loss.backward()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        #acc = logits.detach().max(1)[1] == y
        acc = self.compute_accuracy(logits,y,use_target_indices,selection_mask)

        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()


        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        
        counter3 = 0

        loss_best_last_check = loss_best.clone()

        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            #print('Step',i)
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    
                x_adv = x_adv_1 + 0.

            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                
                with torch.enable_grad():
                    logits = self.model(x_adv,None, None)[0]
                    loss_indiv = -1*mycriterion(logits, y.float())
                    #loss_indiv=torch.mul(loss_indiv,selection_mask).sum(1)
                    
                    loss_indiv = torch.sum(loss_indiv,1)
                    loss = loss_indiv.sum()

                    grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            
            grad /= float(self.eot_iter)

            pred = self.compute_accuracy(logits.detach(),y,use_target_indices,selection_mask)
            # print('Loss:',loss_indiv)
            # print(acc)
            #l=[]

            # for i in range(x_adv.shape[0]):
            #     l.append(torch.max(torch.abs(x_adv-x)))

            #print(l)

            #pred = logits.detach().max(1)[1] == y

            acc = torch.max(acc, pred)
            acc_steps[i + 1] = acc + 0
            #ind_pred = (pred == 0).nonzero().squeeze() hassan
            ind_pred = (pred == 1).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            # if self.verbose:
            #     str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
            #         step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
            #     print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
            #         i, loss_best.sum(), acc.float().mean(), str_stats))
            #     #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1 + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              #print('Loss best:',loss_best)
              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  counter3 = 0
                  k = max(k - self.size_decr, self.n_iter_min)
        #
        #print('\nLoss steps:\n',loss_steps)
        return (x_best, acc, loss_best, x_best_adv)


    def compute_accuracy(self,y_pred,y,use_target_indices,selection_mask):
        # make it true or false. If the goal is achieved, it is true. Elese false
        #hassan compute accuracy for loss1, loss2, loss3, loss4 success
        logits=y_pred.detach()
        logits=torch.where(logits>0,1,0)
        selection_mask=selection_mask.cuda()
        
        if(use_target_indices is True):
            out=torch.mul(torch.eq(logits,y),selection_mask).sum(1)
            out=torch.eq(out,selection_mask.sum(1))
        else:
            out=(torch.eq(logits,y)+0).sum(1)
            out=torch.eq(out,torch.ones_like(selection_mask).sum(1))
        
        
        #out=out.reshape(1,-1)

        return out

    def perturb(self, x, y, mycriterion, use_target_indices, selection_mask, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """
        #selection_mask=torch.from_numpy(selection_mask)
        #torch.save(selection_mask,'selection_mask.pt') #hassan
        #torch.save(x,'X.pt') #hassan
        #torch.save(y,'y.pt') #hassan
        assert self.loss in ['ce', 'dlr'] #'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        
        y_pred = self.model(x,None, None)[0]
        adv = x.clone()
        acc = self.compute_accuracy(y_pred,y,use_target_indices,selection_mask)
        acc=torch.zeros_like(acc) #hassan
        #loss = -1e10 * torch.ones_like(acc).float() #hassan
        loss = -1e10 * torch.ones_like(acc).float()
        
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        
        
        startt = time.time()

        if not best_loss:
            #print('Not best loss')
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = (acc==0).nonzero().squeeze()
                #print('Indices to fool:',ind_to_fool)

                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool,mycriterion,use_target_indices,selection_mask)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr

                    # for l in range(adv_curr.shape[0]):
                    #     print(torch.min(torch.abs(x_to_fool[l]-adv_curr[l])),torch.max(torch.abs(x_to_fool[l]-adv_curr[l])))   
                    
                    #print(torch.min(torch.abs(x_to_fool-adv_curr)),torch.max(torch.abs(x_to_fool-adv_curr)))
                    #ind_curr = (acc_curr == 0).nonzero().squeeze() hassan
                    ind_curr = (acc_curr == 1).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            return adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                # if self.verbose:
                #     print('restart {} - loss: {:.5f}'.format(
                #         counter, loss_best.sum()))

            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        0/0
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)

# class APGDAttack_targeted(APGDAttack):
#     def __init__(
#             self,
#             predict,
#             n_iter=100,
#             norm='Linf',
#             n_restarts=1,
#             eps=None,
#             seed=0,
#             eot_iter=1,
#             rho=.75,
#             topk=None,
#             n_target_classes=9,
#             verbose=False,
#             device=None,
#             use_largereps=False,
#             is_tf_model=False,
#             logger=None):
#         """
#         AutoPGD on the targeted DLR loss
#         """
#         super(APGDAttack_targeted, self).__init__(predict, n_iter=n_iter, norm=norm,
#             n_restarts=n_restarts, eps=eps, seed=seed, loss='dlr-targeted',
#             eot_iter=eot_iter, rho=rho, topk=topk, verbose=verbose, device=device,
#             use_largereps=use_largereps, is_tf_model=is_tf_model, logger=logger)

#         print('\n\n\nHere')
#         self.y_target = None
#         self.n_target_classes = n_target_classes

#     def dlr_loss_targeted(self, x, y):
#         x_sorted, ind_sorted = x.sort(dim=1)
#         u = torch.arange(x.shape[0])

#         return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - .5 * (
#             x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

#     def ce_loss_targeted(self, x, y):
#         return -1. * F.cross_entropy(x, self.y_target, reduction='none')
    
    
#     def perturb(self, x, y=None, x_init=None):
#         """
#         :param x:           clean images
#         :param y:           clean labels, if None we use the predicted labels
#         """

#         assert self.loss in ['dlr-targeted'] #'ce-targeted'
#         if not y is None and len(y.shape) == 0:
#             x.unsqueeze_(0)
#             y.unsqueeze_(0)
#         self.init_hyperparam(x)

#         x = x.detach().clone().float().to(self.device)
#         if not self.is_tf_model:
#             y_pred = self.model(x).max(1)[1]
#         else:
#             y_pred = self.model.predict(x).max(1)[1]
#         if y is None:
#             #y_pred = self._get_predicted_label(x)
#             y = y_pred.detach().clone().long().to(self.device)
#         else:
#             y = y.detach().clone().long().to(self.device)

#         adv = x.clone()
#         acc = y_pred == y
#         if self.verbose:
#             print('-------------------------- ',
#                 'running {}-attack with epsilon {:.5f}'.format(
#                 self.norm, self.eps),
#                 '--------------------------')
#             print('initial accuracy: {:.2%}'.format(acc.float().mean()))

#         startt = time.time()

#         torch.random.manual_seed(self.seed)
#         torch.cuda.random.manual_seed(self.seed)

#         #
        
#         if self.use_largereps:
#             epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
#             iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
#                 .4 * self.n_iter_orig]
#             iters = [math.ceil(c) for c in iters]
#             iters[-1] = self.n_iter_orig - sum(iters[:-1])
#             if self.verbose:
#                 print('using schedule [{}x{}]'.format('+'.join([str(c
#                     ) for c in epss]), '+'.join([str(c) for c in iters])))
        
#         for target_class in range(2, self.n_target_classes + 2):
#             for counter in range(self.n_restarts):
#                 ind_to_fool = acc.nonzero().squeeze()
#                 print(ind_to_fool)
#                 0/0
#                 if len(ind_to_fool.shape) == 0:
#                     ind_to_fool = ind_to_fool.unsqueeze(0)
#                 if ind_to_fool.numel() != 0:
#                     x_to_fool = x[ind_to_fool].clone()
#                     y_to_fool = y[ind_to_fool].clone()
                    
#                     if not self.is_tf_model:
#                         output = self.model(x_to_fool)
#                     else:
#                         output = self.model.predict(x_to_fool)
#                     self.y_target = output.sort(dim=1)[1][:, -target_class]

#                     if not self.use_largereps:
#                         res_curr = self.attack_single_run(x_to_fool, y_to_fool)
#                     else:
#                         res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
#                     best_curr, acc_curr, loss_curr, adv_curr = res_curr
#                     ind_curr = (acc_curr == 0).nonzero().squeeze()

#                     acc[ind_to_fool[ind_curr]] = 0
#                     adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
#                     if self.verbose:
#                         print('target class {}'.format(target_class),
#                             '- restart {} - robust accuracy: {:.2%}'.format(
#                             counter, acc.float().mean()),
#                             '- cum. time: {:.1f} s'.format(
#                             time.time() - startt))

#         return adv

# # Copyright (c) 2020-present, Francesco Croce
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree
# #

# import time
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import random

# from autoattack.other_utils import L0_norm, L1_norm, L2_norm
# from autoattack.checks import check_zero_gradients


# def L1_projection(x2, y2, eps1):
#     '''
#     x2: center of the L1 ball (bs x input_dim)
#     y2: current perturbation (x2 + y2 is the point to be projected)
#     eps1: radius of the L1 ball
#     output: delta s.th. ||y2 + delta||_1 <= eps1
#     and 0 <= x2 + y2 + delta <= 1
#     '''

#     x = x2.clone().float().view(x2.shape[0], -1)
#     y = y2.clone().float().view(y2.shape[0], -1)
#     sigma = y.clone().sign()
#     u = torch.min(1 - x - y, x + y)
#     #u = torch.min(u, epsinf - torch.clone(y).abs())
#     u = torch.min(torch.zeros_like(y), u)
#     l = -torch.clone(y).abs()
#     d = u.clone()
    
#     bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
#     bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
#     inu = 2*(indbs < u.shape[1]).float() - 1
#     size1 = inu.cumsum(dim=1)
    
#     s1 = -u.sum(dim=1)
    
#     c = eps1 - y.clone().abs().sum(dim=1)
#     c5 = s1 + c < 0
#     c2 = c5.nonzero().squeeze(1)
    
#     s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
#     if c2.nelement != 0:
    
#       lb = torch.zeros_like(c2).float()
#       ub = torch.ones_like(lb) *(bs.shape[1] - 1)
      
#       #print(c2.shape, lb.shape)
      
#       nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
#       counter2 = torch.zeros_like(lb).long()
#       counter = 0
          
#       while counter < nitermax:
#         counter4 = torch.floor((lb + ub) / 2.)
#         counter2 = counter4.type(torch.LongTensor)
        
#         c8 = s[c2, counter2] + c[c2] < 0
#         ind3 = c8.nonzero().squeeze(1)
#         ind32 = (~c8).nonzero().squeeze(1)
#         #print(ind3.shape)
#         if ind3.nelement != 0:
#             lb[ind3] = counter4[ind3]
#         if ind32.nelement != 0:
#             ub[ind32] = counter4[ind32]
        
#         #print(lb, ub)
#         counter += 1
        
#       lb2 = lb.long()
#       alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
#       d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
#     return (sigma * d).view(x2.shape)





# class APGDAttack():
#     """
#     AutoPGD
#     https://arxiv.org/abs/2003.01690
#     :param predict:       forward pass function
#     :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
#     :param n_restarts:    number of random restarts
#     :param n_iter:        number of iterations
#     :param eps:           bound on the norm of perturbations
#     :param seed:          random seed for the starting point
#     :param loss:          loss to optimize ('ce', 'dlr' supported)
#     :param eot_iter:      iterations for Expectation over Trasformation
#     :param rho:           parameter for decreasing the step size
#     """

#     def __init__(
#             self,
#             predict,
#             n_iter=1,
#             norm='Linf',
#             n_restarts=1,
#             eps=1e-1,
#             seed=0,
#             loss='ce',
#             eot_iter=1,
#             rho=.75,
#             topk=None,
#             verbose=False,
#             device=None,
#             use_largereps=False,
#             is_tf_model=False,
#             logger=None):
#         """
#         AutoPGD implementation in PyTorch
#         """
        
#         self.model = predict
#         self.n_iter = n_iter
#         self.eps = eps
#         self.norm = norm
#         self.n_restarts = n_restarts
#         self.seed = seed
#         self.loss = loss
#         self.eot_iter = eot_iter
#         self.thr_decr = rho
#         self.topk = topk
#         self.verbose = verbose
#         self.device = device
#         self.use_rs = True
#         #self.init_point = None
#         self.use_largereps = use_largereps
#         #self.larger_epss = None
#         #self.iters = None
#         self.n_iter_orig = n_iter + 0
#         self.eps_orig = eps + 0.
#         self.is_tf_model = is_tf_model
#         self.y_target = None
#         self.logger = logger
    
#     def init_hyperparam(self, x):
#         assert self.norm in ['Linf', 'L2', 'L1']
#         assert not self.eps is None

#         if self.device is None:
#             self.device = x.device
#         self.orig_dim = list(x.shape[1:])
#         self.ndims = len(self.orig_dim)
#         if self.seed is None:
#             self.seed = time.time()
        
#         ### set parameters for checkpoints
#         self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
#         self.n_iter_min = max(int(0.06 * self.n_iter), 1)
#         self.size_decr = max(int(0.03 * self.n_iter), 1)
    
#     def check_oscillation(self, x, j, k, y5, k3=0.75):
#         t = torch.zeros(x.shape[1]).to(self.device)
#         for counter5 in range(k):
#           t += (x[j - counter5] > x[j - counter5 - 1]).float()

#         return (t <= k * k3 * torch.ones_like(t)).float()

#     def check_shape(self, x):
#         return x if len(x.shape) > 0 else x.unsqueeze(0)

#     def normalize(self, x):
#         if self.norm == 'Linf':
#             t = x.abs().view(x.shape[0], -1).max(1)[0]
#             return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

#         elif self.norm == 'L2':
#             t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
#             return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

#         elif self.norm == 'L1':
#             try:
#                 t = x.abs().view(x.shape[0], -1).sum(dim=-1)
#             except:
#                 t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
#             return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
    
#     def lp_norm(self, x):
#         if self.norm == 'L2':
#             t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
#             return t.view(-1, *([1] * self.ndims))

#     def dlr_loss(self, x, y):
#         x_sorted, ind_sorted = x.sort(dim=1)
#         ind = (ind_sorted[:, -1] == y).float()
#         u = torch.arange(x.shape[0])

#         return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
#             1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

#     #
    
#     def attack_single_run(self, x, y, mycriterion, use_target_indices,selection_mask, x_init=None):
        
#         if len(x.shape) < self.ndims:
#             x = x.unsqueeze(0)
#             y = y.unsqueeze(0)

#         if self.norm == 'Linf':
#             t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
#             x_adv = x + self.eps * torch.ones_like(x
#                 ).detach() * self.normalize(t)
        
        
#         if not x_init is None:
#             x_adv = x_init.clone()
            
#         # would be interesting but out of the focus. Thank you for bringing this to attention. we are oigng to add this to paper. 
#         # Please notice that we implemented PGD known to be stronger attack[cite papers] as shown in literature. MIM is related to 
#         # PGD. Relative performance is similar. FGSM we observe 
#         # Thank you and we will add in explanation. 
#         # Explanation
#         # may be more like a suggestion. 
#         # region proposed
#         # we are focussing on these issues. why other context-aware are not but challenges are context-aware in our seeting is this
#         # this would be a subject of different paper. We could not address in a single paper. Why not applicable?
#         # uyes, we can leverage we thought about it and have ideas about it. How we integrate our method?
        
#         x_adv = x_adv.clamp(0., 1.)
#         x_best = x_adv.clone()
#         x_best_adv = x_adv.clone()
#         loss_steps = torch.zeros([self.n_iter, x.shape[0]]
#             ).to(self.device)
#         loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
#             ).to(self.device)
#         acc_steps = torch.zeros_like(loss_best_steps)

        
#         x_adv.requires_grad_()
#         grad = torch.zeros_like(x)
#         for _ in range(self.eot_iter):
#             with torch.enable_grad():
#                 #logits = self.model(x_adv)[0]
#                 logits=self.model(x_adv,None,None)[0]
#                 #logits=self.model(x_adv,epsilon_norm=0.0,target_label=y,target_class=torch.Tensor([0]).long())[0]
#                 loss_indiv = -1*mycriterion(logits, y.type(torch.cuda.FloatTensor))
#                 loss_indiv=torch.mul(loss_indiv,selection_mask).sum(1)


#                 #loss_indiv = torch.sum(loss_indiv,1)
#                 #loss_indiv = criterion_indiv(logits, y)
#                 loss = loss_indiv.sum()

#                 #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
#                 #hassan * -1
#                 #loss.backward()
#                 grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        
        
#         grad /= float(self.eot_iter)
#         grad_best = grad.clone()

#         #acc = logits.detach().max(1)[1] == y
#         acc = self.compute_accuracy(logits,y,use_target_indices,selection_mask)

#         acc_steps[0] = acc + 0
#         loss_best = loss_indiv.detach().clone()


#         alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
#         step_size = alpha * self.eps * torch.ones([x.shape[0], *(
#             [1] * self.ndims)]).to(self.device).detach()
        
#         x_adv_old = x_adv.clone()
#         counter = 0
#         k = self.n_iter_2 + 0
        
#         counter3 = 0

#         loss_best_last_check = loss_best.clone()

#         reduced_last_check = torch.ones_like(loss_best)
#         n_reduced = 0

#         n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
#         u = torch.arange(x.shape[0], device=self.device)
#         for i in range(self.n_iter):
#             #print('Step',i)
#             ### gradient step

#             with torch.no_grad():
#                 x_adv = x_adv.detach()
#                 grad2 = x_adv - x_adv_old
#                 x_adv_old = x_adv.clone()

#                 a = 0.75 if i > 0 else 1.0

#                 if self.norm == 'Linf':
#                     x_adv_1 = x_adv + step_size * torch.sign(grad)
#                     x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
#                         x - self.eps), x + self.eps), 0.0, 1.0)
#                     x_adv_1 = torch.clamp(torch.min(torch.max(
#                         x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
#                         x - self.eps), x + self.eps), 0.0, 1.0)
                    
#                 x_adv = x_adv_1 + 0.

#             ### get gradient
#             x_adv.requires_grad_()
#             grad = torch.zeros_like(x)
#             for _ in range(self.eot_iter):
                
#                 with torch.enable_grad():
#                     #logits = self.model(x_adv)[0]
#                     logits=self.model(x_adv,None,None)[0]
#                     #logits=self.model(x_adv,epsilon_norm=0.0,target_label=y,target_class=torch.Tensor([0]).long())[0]
#                     loss_indiv = -1*mycriterion(logits, y.float())
#                     loss_indiv=torch.mul(loss_indiv,selection_mask).sum(1)
#                     #loss_indiv = torch.sum(loss_indiv,1)
#                     loss = loss_indiv.sum()

#                     grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            
#             #print('Loss:',loss)
#             grad /= float(self.eot_iter)

#             pred = self.compute_accuracy(logits.detach(),y,use_target_indices,selection_mask)
#             # print('Loss:',loss_indiv)
#             # print(acc)
#             #l=[]

#             # for i in range(x_adv.shape[0]):
#             #     l.append(torch.max(torch.abs(x_adv-x)))

#             #print(l)

#             #pred = logits.detach().max(1)[1] == y

#             acc = torch.max(acc, pred)
#             acc_steps[i + 1] = acc + 0
#             #ind_pred = (pred == 0).nonzero().squeeze() hassan
#             ind_pred = (pred == 1).nonzero().squeeze()
#             x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
#             # if self.verbose:
#             #     str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
#             #         step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
#             #     print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
#             #         i, loss_best.sum(), acc.float().mean(), str_stats))
#             #     #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
#             ### check step size
#             with torch.no_grad():
#               y1 = loss_indiv.detach().clone()
#               loss_steps[i] = y1 + 0
#               ind = (y1 > loss_best).nonzero().squeeze()
#               x_best[ind] = x_adv[ind].clone()
#               grad_best[ind] = grad[ind].clone()
#               loss_best[ind] = y1[ind] + 0
#               loss_best_steps[i + 1] = loss_best + 0
#               #print('Loss best:',loss_best)
#               counter3 += 1

#               if counter3 == k:
#                   if self.norm in ['Linf', 'L2']:
#                       fl_oscillation = self.check_oscillation(loss_steps, i, k,
#                           loss_best, k3=self.thr_decr)
#                       fl_reduce_no_impr = (1. - reduced_last_check) * (
#                           loss_best_last_check >= loss_best).float()
#                       fl_oscillation = torch.max(fl_oscillation,
#                           fl_reduce_no_impr)
#                       reduced_last_check = fl_oscillation.clone()
#                       loss_best_last_check = loss_best.clone()
    
#                       if fl_oscillation.sum() > 0:
#                           ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
#                           step_size[ind_fl_osc] /= 2.0
#                           n_reduced = fl_oscillation.sum()
    
#                           x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
#                           grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

#                       k = max(k - self.size_decr, self.n_iter_min)
                  
#                   counter3 = 0
#                   k = max(k - self.size_decr, self.n_iter_min)
#         #
#         #print('\nLoss steps:\n',loss_steps)
#         return (x_best, acc, loss_best, x_best_adv)


#     def compute_accuracy(self,y_pred,y,use_target_indices,selection_mask):
#         # make it true or false. If the goal is achieved, it is true. Elese false
#         #hassan compute accuracy for loss1, loss2, loss3, loss4 success
#         logits=y_pred.detach()
#         logits=torch.where(logits>0,1,0)
#         selection_mask=selection_mask.cuda()
        
#         if(use_target_indices is True):
#             out=torch.mul(torch.eq(logits,y),selection_mask).sum(1)
#             out=torch.eq(out,selection_mask.sum(1))
#         else:
#             out=(torch.eq(logits,y)+0).sum(1)
#             out=torch.eq(out,torch.ones_like(selection_mask).sum(1))
        
        
#         #out=out.reshape(1,-1)

#         return out

#     def perturb(self, x, y, mycriterion, use_target_indices, selection_mask, best_loss=False, x_init=None):

#         mycriterion=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')

#         #mycriterion=torch.mul(torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')(outputs,labels),self.selection_mask)
#         """
#         :param x:           clean images
#         :param y:           clean labels, if None we use the predicted labels
#         :param best_loss:   if True the points attaining highest loss
#                             are returned, otherwise adversarial examples
#         """
#         #selection_mask=torch.from_numpy(selection_mask).cuda()
#         selection_mask=selection_mask.cuda()
#         #torch.save(selection_mask,'selection_mask.pt') #hassan
#         #torch.save(x,'X.pt') #hassan
#         #torch.save(y,'y.pt') #hassan
#         assert self.loss in ['ce', 'dlr'] #'ce-targeted-cfts'
#         if not y is None and len(y.shape) == 0:
#             x.unsqueeze_(0)
#             y.unsqueeze_(0)
#         self.init_hyperparam(x)

#         x = x.detach().clone().float().to(self.device)
        
#         y_pred = self.model(x,None,None)[0]#self.model(x,epsilon_norm=0.0,target_label=y,target_class=torch.Tensor([0]).long())[0]
#         adv = x.clone()
#         acc = self.compute_accuracy(y_pred,y,use_target_indices,selection_mask)
#         acc=torch.zeros_like(acc) #hassan
#         #loss = -1e10 * torch.ones_like(acc).float() #hassan
#         loss = -1e10 * torch.ones_like(acc).float()
        
#         if self.verbose:
#             print('-------------------------- ',
#                 'running {}-attack with epsilon {:.5f}'.format(
#                 self.norm, self.eps),
#                 '--------------------------')
#             print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        
        
#         startt = time.time()

#         if not best_loss:
#             #print('Not best loss')
#             torch.random.manual_seed(self.seed)
#             torch.cuda.random.manual_seed(self.seed)

#             for counter in range(self.n_restarts):
#                 ind_to_fool = (acc==0).nonzero().squeeze()
#                 #print('Indices to fool:',ind_to_fool)

#                 if len(ind_to_fool.shape) == 0:
#                     ind_to_fool = ind_to_fool.unsqueeze(0)
#                 if ind_to_fool.numel() != 0:
#                     x_to_fool = x[ind_to_fool].clone()
#                     y_to_fool = y[ind_to_fool].clone()
                    
#                     if not self.use_largereps:
#                         res_curr = self.attack_single_run(x_to_fool, y_to_fool,mycriterion,use_target_indices,selection_mask)
#                     else:
#                         res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
#                     best_curr, acc_curr, loss_curr, adv_curr = res_curr

#                     # for l in range(adv_curr.shape[0]):
#                     #     print(torch.min(torch.abs(x_to_fool[l]-adv_curr[l])),torch.max(torch.abs(x_to_fool[l]-adv_curr[l])))   
                    
#                     #print(torch.min(torch.abs(x_to_fool-adv_curr)),torch.max(torch.abs(x_to_fool-adv_curr)))
#                     #ind_curr = (acc_curr == 0).nonzero().squeeze() hassan
#                     ind_curr = (acc_curr == 1).nonzero().squeeze()

#                     acc[ind_to_fool[ind_curr]] = 0
#                     adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
#                     if self.verbose:
#                         print('restart {} - robust accuracy: {:.2%}'.format(
#                             counter, acc.float().mean()),
#                             '- cum. time: {:.1f} s'.format(
#                             time.time() - startt))

#             return adv

#         else:
#             adv_best = x.detach().clone()
#             loss_best = torch.ones([x.shape[0]]).to(
#                 self.device) * (-float('inf'))
#             for counter in range(self.n_restarts):
#                 best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
#                 ind_curr = (loss_curr > loss_best).nonzero().squeeze()
#                 adv_best[ind_curr] = best_curr[ind_curr] + 0.
#                 loss_best[ind_curr] = loss_curr[ind_curr] + 0.

#                 # if self.verbose:
#                 #     print('restart {} - loss: {:.5f}'.format(
#                 #         counter, loss_best.sum()))

#             return adv_best

#     def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
#         assert len(epss) == len(iters)
#         assert self.norm in ['L1']
#         self.use_rs = False
#         if not use_rs:
#             x_init = None
#         else:
#             x_init = x + torch.randn_like(x)
#             x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
#         eps_target = float(epss[-1])
#         if self.verbose:
#             print('total iter: {}'.format(sum(iters)))
#         for eps, niter in zip(epss, iters):
#             if self.verbose:
#                 print('using eps: {:.2f}'.format(eps))
#             self.n_iter = niter + 0
#             self.eps = eps + 0.
#             #
#             if not x_init is None:
#                 x_init += L1_projection(x, x_init - x, 1. * eps)
#             x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

#         return (x_init, acc, loss, x_adv)

# # class APGDAttack_targeted(APGDAttack):
# #     def __init__(
# #             self,
# #             predict,
# #             n_iter=100,
# #             norm='Linf',
# #             n_restarts=1,
# #             eps=None,
# #             seed=0,
# #             eot_iter=1,
# #             rho=.75,
# #             topk=None,
# #             n_target_classes=9,
# #             verbose=False,
# #             device=None,
# #             use_largereps=False,
# #             is_tf_model=False,
# #             logger=None):
# #         """
# #         AutoPGD on the targeted DLR loss
# #         """
# #         super(APGDAttack_targeted, self).__init__(predict, n_iter=n_iter, norm=norm,
# #             n_restarts=n_restarts, eps=eps, seed=seed, loss='dlr-targeted',
# #             eot_iter=eot_iter, rho=rho, topk=topk, verbose=verbose, device=device,
# #             use_largereps=use_largereps, is_tf_model=is_tf_model, logger=logger)

# #         print('\n\n\nHere')
# #         self.y_target = None
# #         self.n_target_classes = n_target_classes

# #     def dlr_loss_targeted(self, x, y):
# #         x_sorted, ind_sorted = x.sort(dim=1)
# #         u = torch.arange(x.shape[0])

# #         return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - .5 * (
# #             x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)

# #     def ce_loss_targeted(self, x, y):
# #         return -1. * F.cross_entropy(x, self.y_target, reduction='none')
    
    
# #     def perturb(self, x, y=None, x_init=None):
# #         """
# #         :param x:           clean images
# #         :param y:           clean labels, if None we use the predicted labels
# #         """

# #         assert self.loss in ['dlr-targeted'] #'ce-targeted'
# #         if not y is None and len(y.shape) == 0:
# #             x.unsqueeze_(0)
# #             y.unsqueeze_(0)
# #         self.init_hyperparam(x)

# #         x = x.detach().clone().float().to(self.device)
# #         if not self.is_tf_model:
# #             y_pred = self.model(x).max(1)[1]
# #         else:
# #             y_pred = self.model.predict(x).max(1)[1]
# #         if y is None:
# #             #y_pred = self._get_predicted_label(x)
# #             y = y_pred.detach().clone().long().to(self.device)
# #         else:
# #             y = y.detach().clone().long().to(self.device)

# #         adv = x.clone()
# #         acc = y_pred == y
# #         if self.verbose:
# #             print('-------------------------- ',
# #                 'running {}-attack with epsilon {:.5f}'.format(
# #                 self.norm, self.eps),
# #                 '--------------------------')
# #             print('initial accuracy: {:.2%}'.format(acc.float().mean()))

# #         startt = time.time()

# #         torch.random.manual_seed(self.seed)
# #         torch.cuda.random.manual_seed(self.seed)

# #         #
        
# #         if self.use_largereps:
# #             epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
# #             iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig,
# #                 .4 * self.n_iter_orig]
# #             iters = [math.ceil(c) for c in iters]
# #             iters[-1] = self.n_iter_orig - sum(iters[:-1])
# #             if self.verbose:
# #                 print('using schedule [{}x{}]'.format('+'.join([str(c
# #                     ) for c in epss]), '+'.join([str(c) for c in iters])))
        
# #         for target_class in range(2, self.n_target_classes + 2):
# #             for counter in range(self.n_restarts):
# #                 ind_to_fool = acc.nonzero().squeeze()
# #                 print(ind_to_fool)
# #                 0/0
# #                 if len(ind_to_fool.shape) == 0:
# #                     ind_to_fool = ind_to_fool.unsqueeze(0)
# #                 if ind_to_fool.numel() != 0:
# #                     x_to_fool = x[ind_to_fool].clone()
# #                     y_to_fool = y[ind_to_fool].clone()
                    
# #                     if not self.is_tf_model:
# #                         output = self.model(x_to_fool)
# #                     else:
# #                         output = self.model.predict(x_to_fool)
# #                     self.y_target = output.sort(dim=1)[1][:, -target_class]

# #                     if not self.use_largereps:
# #                         res_curr = self.attack_single_run(x_to_fool, y_to_fool)
# #                     else:
# #                         res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
# #                     best_curr, acc_curr, loss_curr, adv_curr = res_curr
# #                     ind_curr = (acc_curr == 0).nonzero().squeeze()

# #                     acc[ind_to_fool[ind_curr]] = 0
# #                     adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
# #                     if self.verbose:
# #                         print('target class {}'.format(target_class),
# #                             '- restart {} - robust accuracy: {:.2%}'.format(
# #                             counter, acc.float().mean()),
# #                             '- cum. time: {:.1f} s'.format(
# #                             time.time() - startt))

# #         return adv