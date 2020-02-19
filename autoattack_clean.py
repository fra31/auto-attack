import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import time

class AutoAttack():
    def __init__(self, model, norm='Linf', eps=.3, seed=0, verbose=True,
                 attacks_to_run=['apgd-ce', 'apgd-dlr', 'fab', 'square']):
        self.model = model
        self.norm = norm
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        
        from autopgd_pt_clean import APGDAttack
        self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, n_iter_2=22, n_iter_min=6, size_decr=3,
            eps=self.epsilon, show_loss=False, norm=self.norm, eot_iter=1, thr_decr=.75, seed=self.seed,
            show_acc=False)
        
        from fab_pt import FABAttack
        self.fab = FABAttack(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed, norm=self.norm,
            verbose=False, verbose_restarts=False)
        
        from square_attack import SquareAttack
        self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            show_progr=False, early_stop=True, n_restarts=1, show_acc=False, seed=self.seed)
            
    def run_standard_evaluation(self, x_orig, y_orig, bs=250):
        with torch.no_grad():
            n_batches = x_orig.shape[0] // bs
            acc_total = 0.
            adv = x_orig.detach().clone().cpu()
            
            for counter in range(n_batches):
                x, y = x_orig[counter * bs:(counter + 1) * bs].clone().cuda(), y_orig[counter * bs:(counter + 1) * bs].clone().cuda()
                x_adv = x.clone()
                
                output = self.model(x)
                acc = (output.max(1)[1] == y).float()
                ind_to_fool = (output.max(1)[1] == y).nonzero().squeeze()
                
                if self.verbose:
                    print('initial accuracy batch {}: {:.1%}'.format(counter + 1, acc.float().mean()))
                    
                # apgd on cross-entropy loss
                if 'apgd-ce' in self.attacks_to_run:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    self.apgd.loss = 'ce'
                    self.apgd.seed = time.time()
                    _, adv_curr = self.apgd.perturb(x_to_fool, y_to_fool, cheap=True)
                    
                    ind_succ = (self.model(adv_curr).max(1)[1] != y[ind_to_fool]).nonzero().squeeze()
                    x_adv[ind_to_fool[ind_succ]] = adv_curr[ind_succ] + 0.
                    acc[ind_to_fool[ind_succ]] = 0.
                    
                    if self.verbose:
                        print('robust accuracy batch {} after APGD-CE: {:.1%}'.format(counter + 1, acc.float().mean()))
                    
                # apgd on DLR loss
                ind_to_fool = (acc == 1.).nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0 and 'apgd-dlr' in self.attacks_to_run:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    self.apgd.loss = 'dlr'
                    self.apgd.seed = time.time()
                    _, adv_curr = self.apgd.perturb(x_to_fool, y_to_fool, cheap=True)
                    
                    ind_succ = (self.model(adv_curr).max(1)[1] != y[ind_to_fool]).nonzero().squeeze()
                    x_adv[ind_to_fool[ind_succ]] = adv_curr[ind_succ] + 0.
                    acc[ind_to_fool[ind_succ]] = 0.
                    
                    if self.verbose:
                        print('robust accuracy batch {} after APGD-DLR: {:.1%}'.format(counter + 1, acc.float().mean()))
                    
                # fab
                ind_to_fool = (acc == 1.).nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0 and 'fab' in self.attacks_to_run:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    self.fab.seed = time.time()
                    adv_curr = self.fab.perturb(x_to_fool, y_to_fool)
                    
                    ind_succ = (self.model(adv_curr).max(1)[1] != y[ind_to_fool]).nonzero().squeeze()
                    x_adv[ind_to_fool[ind_succ]] = adv_curr[ind_succ] + 0.
                    acc[ind_to_fool[ind_succ]] = 0.
                    
                    if self.verbose:
                        print('robust accuracy batch {} after FAB: {:.1%}'.format(counter + 1, acc.float().mean()))
                    
                # square
                ind_to_fool = (acc == 1.).nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0 and 'square' in self.attacks_to_run:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    self.square.seed = time.time()
                    _, adv_curr = self.square.perturb(x_to_fool, y_to_fool)
                    
                    ind_succ = (self.model(adv_curr).max(1)[1] != y[ind_to_fool]).nonzero().squeeze()
                    x_adv[ind_to_fool[ind_succ]] = adv_curr[ind_succ] + 0.
                    acc[ind_to_fool[ind_succ]] = 0.
                    
                    if self.verbose:
                        print('robust accuracy batch {} after Square Attack: {:.1%}'.format(counter + 1, acc.float().mean()))
                    
                res = (x_adv - x).abs().view(x.shape[0], -1).max(1)[0]
                if self.verbose:
                    print('max perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                        res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                adv[counter * bs:(counter + 1) * bs] = x_adv.cpu() + 0.
                acc_total += acc.sum()        
        
        if self.verbose:
            print('robust accuracy: {:.2%}'.format(acc_total / x_orig.shape[0]))
        
        return adv