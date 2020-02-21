import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import time

class AutoAttack():
    def __init__(self, model, norm='Linf', eps=.3, seed=0, verbose=True,
                 attacks_to_run=['apgd-ce', 'apgd-dlr', 'fab', 'square'],
                 plus=False):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run if not plus else attacks_to_run.extend(['apgd-t', 'fab-t'])
        self.plus = plus
        
        from autopgd_pt import APGDAttack
        self.apgd = APGDAttack(self.model, n_restarts=5, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed)
        
        from fab_pt import FABAttack
        self.fab = FABAttack(self.model, n_restarts=5, n_iter=100, eps=self.epsilon, seed=self.seed,
            norm=self.norm, verbose=False)
        
        from square import SquareAttack
        self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            early_stop=True, n_restarts=1, seed=self.seed, verbose=False)
            
        from autopgd_pt import APGDAttack_targeted
        self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed)
    
    def run_standard_evaluation(self, x_orig, y_orig, bs=250):
        # update attacks list if plus activated after initialization
        if self.plus:
            if not 'apgd-t' in self.attacks_to_run:
                self.attacks_to_run.extend(['apgd-t'])
            if not 'fab-t' in self.attacks_to_run:
                self.attacks_to_run.extend(['fab-t'])
        
        with torch.no_grad():
            n_batches = x_orig.shape[0] // bs
            acc_total = 0.
            adv = x_orig.detach().clone().cpu()
            
            for counter in range(n_batches):
                startt = time.time()
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
                        print('robust accuracy batch {} after APGD-CE \t {:.1%} \t (time batch: {:.1f} s)'.format(
                            counter + 1, acc.float().mean(), time.time() - startt))
                    
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
                        print('robust accuracy batch {} after APGD-DLR \t {:.1%} \t (time batch: {:.1f} s)'.format(
                            counter + 1, acc.float().mean(), time.time() - startt))
                    
                # fab
                ind_to_fool = (acc == 1.).nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0 and 'fab' in self.attacks_to_run:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    self.fab.targeted = False
                    self.fab.seed = time.time()
                    adv_curr = self.fab.perturb(x_to_fool, y_to_fool)
                    
                    ind_succ = (self.model(adv_curr).max(1)[1] != y[ind_to_fool]).nonzero().squeeze()
                    x_adv[ind_to_fool[ind_succ]] = adv_curr[ind_succ] + 0.
                    acc[ind_to_fool[ind_succ]] = 0.
                    
                    if self.verbose:
                        print('robust accuracy batch {} after FAB \t {:.1%} \t (time batch: {:.1f} s)'.format(
                            counter + 1, acc.float().mean(), time.time() - startt))
                    
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
                        print('robust accuracy batch {} after Square \t {:.1%} \t (time batch: {:.1f} s)'.format(
                            counter + 1, acc.float().mean(), time.time() - startt))
                    
                # apgd targeted
                ind_to_fool = (acc == 1.).nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0 and 'apgd-t' in self.attacks_to_run:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    self.apgd_targeted.seed = time.time()
                    _, adv_curr = self.apgd_targeted.perturb(x_to_fool, y_to_fool, cheap=True)
                    
                    ind_succ = (self.model(adv_curr).max(1)[1] != y[ind_to_fool]).nonzero().squeeze()
                    x_adv[ind_to_fool[ind_succ]] = adv_curr[ind_succ] + 0.
                    acc[ind_to_fool[ind_succ]] = 0.
                    
                    if self.verbose:
                        print('robust accuracy batch {} after APGD-T \t {:.1%} \t (time batch: {:.1f} s)'.format(
                            counter + 1, acc.float().mean(), time.time() - startt))
                
                # fab targeted
                ind_to_fool = (acc == 1.).nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0 and 'fab-t' in self.attacks_to_run:
                    x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                    self.fab.targeted = True
                    self.fab.seed = time.time()
                    adv_curr = self.fab.perturb(x_to_fool, y_to_fool)
                    
                    ind_succ = (self.model(adv_curr).max(1)[1] != y[ind_to_fool]).nonzero().squeeze()
                    x_adv[ind_to_fool[ind_succ]] = adv_curr[ind_succ] + 0.
                    acc[ind_to_fool[ind_succ]] = 0.
                    
                    if self.verbose:
                        print('robust accuracy batch {} after FAB-T \t {:.1%} \t (time batch: {:.1f} s)'.format(
                            counter + 1, acc.float().mean(), time.time() - startt))
                
                adv[counter * bs:(counter + 1) * bs] = x_adv.cpu() + 0.
                acc_total += acc.sum()        
                
        # final check
        if self.verbose:
            if self.norm == 'Linf':
                res = (adv - x_orig).abs().view(x_orig.shape[0], -1).max(1)[0]
            elif self.norm == 'L2':
                res = ((adv - x_orig) ** 2).view(x_orig.shape[0], -1).sum(-1).sqrt()
            print('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                self.norm, res.max(), (adv != adv).sum(), adv.max(), adv.min()))
            print('robust accuracy: {:.2%}'.format(acc_total / x_orig.shape[0]))
        
        return adv
        
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = x_orig.shape[0] // bs
        acc = 0.
        for counter in range(n_batches):
            x, y = x_orig[counter * bs:(counter + 1) * bs].clone().cuda(), y_orig[counter * bs:(counter + 1) * bs].clone().cuda()
            output = self.model(x)
            acc += (output.max(1)[1] == y).float().sum()
            
        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))
        
        return acc.item() / x_orig.shape[0]
        
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250):
        # update attacks list if plus activated after initialization
        if self.plus:
            if not 'apgd-t' in self.attacks_to_run:
                self.attacks_to_run.extend(['apgd-t'])
            if not 'fab-t' in self.attacks_to_run:
                self.attacks_to_run.extend(['fab-t'])
        
        l_attacks = self.attacks_to_run
        adv = {}
        self.plus = False
        verbose_indiv = self.verbose
        self.verbose = False
        
        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            adv[c] = self.run_standard_evaluation(x_orig, y_orig, bs=bs)
            if verbose_indiv:    
                acc_indiv  = self.clean_accuracy(adv[c], y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                print('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(
                    c.upper(), space, acc_indiv,  time.time() - startt))
        
        return adv