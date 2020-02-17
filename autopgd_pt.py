#import tensorflow as tf
import numpy as np
import time
import torch
import scipy.io

import numpy.linalg as nl

#import settings
import os
import sys

import torch.nn as nn
import torch.nn.functional as F
#from torch.autograd import Variable
    
class APGDAttack_singlestepsize():
    def __init__(self, model, n_iter=100, n_iter_2=18, n_iter_min=6, size_decr=3,
                 norm='Linf', n_restarts=1, eps=0.3, show_loss=False, seed=0,
                 loss='ce', show_acc=True, eot_iter=1, save_steps=False,
                 save_dir='./results/', thr_decr=.75, check_impr=False,
                 normalize_logits=False):
        self.model = model
        self.n_iter = n_iter
        self.n_iter_2 = n_iter_2
        self.n_iter_min = n_iter_min
        self.size_decr = size_decr
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.show_loss = show_loss    
        self.seed = seed
        self.loss = loss
        self.show_acc = show_acc
        self.eot_iter = eot_iter
        self.save_steps = save_steps    
        self.save_dir = save_dir
        self.thr_decr = thr_decr
        self.check_impr = check_impr
        self.normalize_logits = normalize_logits
    
    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]
          
        return t <= k*k3*np.ones(t.shape), t > k*1.0*np.ones(t.shape)
        
    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)
    
    def norm_to_interval(self, x):
        return x / (x.max(dim=1, keepdim=True)[0] + 1e-12)
        
    
    def custom_loss(self, x, y=None):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        
        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    
    def attack_single_run(self, x_in, y_in):
        x = x_in if len(x_in.shape) == 4 else x_in.unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).cuda().detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).cuda().detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        
        if self.loss == 'ce':
            criterion = nn.CrossEntropyLoss(size_average=False)
            criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'kl_div':
            criterion = nn.KLDivLoss(size_average=False)
            criterion_indiv = nn.KLDivLoss(reduce=False, reduction='none')
        #elif self.loss =='custom':
        #    criterion_indiv = self.custom_loss
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                if self.loss == 'kl_div':
                    loss = criterion(F.log_softmax(self.model(x_adv), dim=1), F.softmax(self.model(x), dim=1))
                else:
                    if not self.normalize_logits:
                        logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                        loss_indiv = criterion_indiv(logits, y)
                        loss = loss_indiv.sum()
                    else:
                        #loss = self.custom_loss(self.model(x_adv), y).sum()
                        logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                        loss_indiv = self.custom_loss(logits, y)
                        loss = loss_indiv.sum()
            
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        
        loss_best = loss_indiv.detach().clone()
        loss = loss_best.sum()
        
        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * torch.Tensor([2.0]).cuda().detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        a = 0.75
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        #if self.loss == 'kl_div': loss = criterion(F.log_softmax(self.model(x_adv), dim=1), F.softmax(self.model(x), dim=1))
        #else: loss = criterion(self.model(x_adv), y)
        #grad2 = torch.zeros([1])
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                    
                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                    
                x_adv = x_adv_1 + 0.
            
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    if self.loss == 'kl_div':
                        loss = criterion(F.log_softmax(self.model(x_adv), dim=1), F.softmax(self.model(x), dim=1))
                    else:
                        if not self.normalize_logits:
                            logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                            loss_indiv = criterion_indiv(logits, y)
                            loss = loss_indiv.sum()
                        else:
                            #loss = self.custom_loss(self.model(x_adv), y).sum()
                            logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                            loss_indiv = self.custom_loss(logits, y)
                            loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            
            grad /= float(self.eot_iter)
            
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
            if self.show_loss: print('iteration: {} - Best loss: {:.6f} - Step size: {:.4f} - Reduced: {:.0f}'.format(i, loss_best.sum(), step_size.mean(), n_reduced))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation, _ = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                  
                  if self.check_impr:
                      fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                      fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                      reduced_last_check = np.copy(fl_oscillation)
                      loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                      '''x_new = x_best[fl_oscillation].clone().requires_grad_()
                      y_new = y[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                      with torch.enable_grad():  
                          grad_new = torch.zeros_like(x_new)
                          for _ in range(self.eot_iter):
                              if self.loss == 'kl_div':
                                  #loss = criterion(F.log_softmax(self.model(x_new), dim=1), F.softmax(self.model(x), dim=1))
                                  raise ValueError('not implemented yet')
                              else:
                                  if not self.normalize_logits:
                                      logits = self.model(x_new) # 1 forward pass (eot_iter = 1)
                                      loss_indiv = criterion_indiv(logits, y_new)
                                      loss = loss_indiv.sum()
                                  else:
                                      loss = self.custom_loss(self.model(x_new), y_new).sum()
                      
                              grad_new += torch.autograd.grad(loss, [x_new])[0].detach() # 1 backward pass (eot_iter = 1)
                      
                      grad[fl_oscillation] = grad_new / float(self.eot_iter)'''
                  
                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.n_iter_min)
              
        ### save intermediate steps 
        if self.save_steps:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            return acc_steps, loss_best_steps
            
            torch.save({'acc_steps': acc_steps, 'loss_steps': loss_best_steps}, self.save_dir + '/apgd_singlestep_{}_eps_{:.5f}_niter_{:.0f}_thrdecr_{:.2}.pth'.format(
                self.norm, self.eps, self.n_iter, self.thr_decr))
            scipy.io.savemat(self.save_dir + '/apgd_singlestep_{}_eps_{:.5f}_niter_{:.0f}_thrdecr_{:.2}.pth'.format(
                self.norm, self.eps, self.n_iter, self.thr_decr), {'acc_steps': acc_steps.cpu().detach().numpy(), 'loss_steps': loss_best_steps.cpu().detach().numpy()})
        
        return x_best, acc, loss_best, x_best_adv
    
    def perturb(self, x, y, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        
        adv = x.clone()
        acc = self.model(x).max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.show_acc:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        
        if self.save_steps:
            assert self.n_restarts == 1
            acc, loss = self.attack_single_run(x, y)
            
            return acc, loss
        
        if not cheap:
            for counter in range(self.n_restarts):
                if best_loss:
                    ### to do: return highest loss
                    best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x, y)
                    ind = (acc_curr == 0) #(loss_curr >= loss)
                    ind = ind.nonzero().squeeze()
                    acc = torch.min(acc, acc_curr)
                    loss = torch.max(loss, loss_curr)
                    adv[ind] = adv_curr[ind].clone()
                    adv = adv * (acc_curr == 1).float().view(-1, 1, 1, 1) + adv_curr * (acc_curr == 0).float().view(-1, 1, 1, 1)
                    if self.show_acc: print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(counter, acc.float().mean(), time.time() - startt))
                
                else:
                    best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x, y)
                    ind = (acc_curr == 0) #(loss_curr >= loss)
                    ind = ind.nonzero().squeeze()
                    acc = torch.min(acc, acc_curr)
                    loss = torch.max(loss, loss_curr)
                    adv[ind] = adv_curr[ind].clone()
                    adv = adv * (acc_curr == 1).float().view(-1, 1, 1, 1) + adv_curr * (acc_curr == 0).float().view(-1, 1, 1, 1)
                    if self.show_acc: print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(counter, acc.float().mean(), time.time() - startt))
                    #output = self.model(adv)    
                    #print('test adversarials across restarts: {:.2%}'.format((output.max(1)[1] == y).float().mean()))
        
        else:
            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                    #acc_temp = torch.zeros_like(acc)
                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.show_acc: print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(counter, acc.float().mean(), time.time() - startt))
        
        #y_pred = (self.model(adv).max(1)[1] == y).float()
        #print(y_pred.mean()*100, (adv - x).abs().reshape([-1]).max(0)[0])
        
        return acc, adv
        
class APGDAttack_targeted():
    def __init__(self, model, n_iter=100, n_iter_2=22, n_iter_min=6, size_decr=3,
                 norm='Linf', n_restarts=1, eps=0.3, show_loss=False, seed=0,
                 loss='ce', show_acc=True, eot_iter=1, save_steps=False,
                 save_dir='./results/', thr_decr=.75, check_impr=True,
                 normalize_logits=False, target_class=None):
        self.model = model
        self.n_iter = n_iter
        self.n_iter_2 = n_iter_2
        self.n_iter_min = n_iter_min
        self.size_decr = size_decr
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.show_loss = show_loss    
        self.seed = seed
        self.loss = loss
        self.show_acc = show_acc
        self.eot_iter = eot_iter
        self.save_steps = save_steps    
        self.save_dir = save_dir
        self.thr_decr = thr_decr
        self.check_impr = check_impr
        self.normalize_logits = normalize_logits
        self.target_class = target_class
    
    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]
          
        return t <= k*k3*np.ones(t.shape), t > k*1.0*np.ones(t.shape)
        
    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)
    
    def custom_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = x.sort(dim=1)
        #ind = (ind_sorted[:, -1] == y).float()
        
        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (x_sorted[:, -1] - .5 * x_sorted[:, -3] - .5 * x_sorted[:, -4] + 1e-12)
    
    def attack_single_run(self, x_in, y_in):
        x = x_in if len(x_in.shape) == 4 else x_in.unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        
        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).cuda().detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).cuda().detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e12)
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)
        
        output = self.model(x)
        y_target = output.sort(dim=1)[1][:, -self.target_class]
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                loss_indiv = self.custom_loss_targeted(logits, y, y_target)
                loss = loss_indiv.sum()
            
            grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        grad /= float(self.eot_iter)
        grad_best = grad.clone()
        
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        
        loss_best = loss_indiv.detach().clone()
        loss = loss_best.sum()
        
        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).cuda().detach() * torch.Tensor([2.0]).cuda().detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        a = 0.75
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        #
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0
        
        for i in range(self.n_iter):
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                    
                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).cuda().detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                    
                x_adv = x_adv_1 + 0.
            
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv) # 1 forward pass (eot_iter = 1)
                    loss_indiv = self.custom_loss_targeted(logits, y, y_target)
                    loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            
            grad /= float(self.eot_iter)
            
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
            if self.show_loss: print('iteration: {} - Best loss: {:.6f} - Step size: {:.4f} - Reduced: {:.0f}'.format(i, loss_best.sum(), step_size.mean(), n_reduced))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation, _ = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                  
                  if self.check_impr:
                      fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                      fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                      reduced_last_check = np.copy(fl_oscillation)
                      loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                  counter3 = 0
                  k = np.maximum(k - self.size_decr, self.n_iter_min)
              
        ### save intermediate steps 
        if self.save_steps:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            
            return acc_steps, loss_best_steps
            
            torch.save({'acc_steps': acc_steps, 'loss_steps': loss_best_steps}, self.save_dir + '/apgd_singlestep_{}_eps_{:.5f}_niter_{:.0f}_thrdecr_{:.2}.pth'.format(
                self.norm, self.eps, self.n_iter, self.thr_decr))
            scipy.io.savemat(self.save_dir + '/apgd_singlestep_{}_eps_{:.5f}_niter_{:.0f}_thrdecr_{:.2}.pth'.format(
                self.norm, self.eps, self.n_iter, self.thr_decr), {'acc_steps': acc_steps.cpu().detach().numpy(), 'loss_steps': loss_best_steps.cpu().detach().numpy()})
        
        return x_best, acc, loss_best, x_best_adv
    
    def perturb(self, x, y, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        
        adv = x.clone()
        acc = self.model(x).max(1)[1] == y
        loss = -1e10 * torch.ones_like(acc).float()
        if self.show_acc:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()
        
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        
        if self.save_steps:
            assert self.n_restarts == 1
            acc, loss = self.attack_single_run(x, y)
            
            return acc, loss
        
        if not cheap:
            raise ValueError('not implemented yet')
        
        else:
            for target_class in range(2, 11):
                self.target_class = target_class
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        #
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.show_acc:
                            print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, self.target_class, acc.float().mean(), self.eps, time.time() - startt))
                        
        return acc, adv
        
