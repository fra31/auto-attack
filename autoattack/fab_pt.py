# Copyright (c) 2019-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time

import torch
from torch.autograd.gradcheck import zero_gradients

from autoattack.fab_projections import projection_linf, projection_l2,\
    projection_l1

DEFAULT_EPS_DICT_BY_NORM = {'Linf': .3, 'L2': 1., 'L1': 5.0}


class FABAttack():
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044
    
    :param predict:       forward pass function
    :param norm:          Lp-norm to minimize ('Linf', 'L2', 'L1' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           epsilon for the random restarts
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    """

    def __init__(
            self,
            predict,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=None,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            loss_fn=None,
            verbose=False,
            seed=0,
            targeted=False,
            device=None,
            n_target_classes=9):
        """ FAB-attack implementation in pytorch """

        self.predict = predict
        self.norm = norm
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.eps = eps if eps is not None else DEFAULT_EPS_DICT_BY_NORM[norm]
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.targeted = False
        self.verbose = verbose
        self.seed = seed
        self.target_class = None
        self.device = device
        self.n_target_classes = n_target_classes

    def _get_predicted_label(self, x):
        with torch.no_grad():
            outputs = self.predict(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.predict(im)

        g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(self.device)
        grad_mask = torch.zeros_like(y)
        for counter in range(y.shape[-1]):
            zero_gradients(im)
            grad_mask[:, counter] = 1.0
            y.backward(grad_mask, retain_graph=True)
            grad_mask[:, counter] = 0.0
            g2[counter] = im.grad.data

        g2 = torch.transpose(g2, 0, 1).detach()
        #y2 = self.predict(imgs).detach()
        y2 = y.detach()
        df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        df[torch.arange(imgs.shape[0]), la] = 1e10

        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        u = torch.arange(imgs.shape[0])
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.predict(im)
            diffy = -(y[u, la] - y[u, la_target])
            sumdiffy = diffy.sum()

        zero_gradients(im)
        sumdiffy.backward()
        graddiffy = im.grad.data
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)

        return df, dg

    def attack_single_run(self, x, y=None, use_rand_start=False):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        #self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)
        #assert next(self.predict.parameters()).device == x.device

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        corr_classified = pred.float().sum()
        if self.verbose:
            print('Clean accuracy: {:.2%}'.format(pred.float().mean()))
        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                if self.norm == 'Linf':
                    t = 2 * torch.rand(x1.shape).to(self.device) - 1
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / (t.reshape([t.shape[0], -1]).abs()
                                         .max(dim=1, keepdim=True)[0]
                                         .reshape([-1, *[1]*self.ndims])) * .5
                elif self.norm == 'L2':
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / ((t ** 2)
                                         .view(t.shape[0], -1)
                                         .sum(dim=-1)
                                         .sqrt()
                                         .view(t.shape[0], *[1]*self.ndims)) * .5
                elif self.norm == 'L1':
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / (t.abs().view(t.shape[0], -1)
                                         .sum(dim=-1)
                                         .view(t.shape[0], *[1]*self.ndims)) / 2

                x1 = x1.clamp(0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.n_iter:
                with torch.no_grad():
                    df, dg = self.get_diff_logits_grads_batch(x1, la2)
                    if self.norm == 'Linf':
                        dist1 = df.abs() / (1e-12 +
                                            dg.abs()
                                            .view(dg.shape[0], dg.shape[1], -1)
                                            .sum(dim=-1))
                    elif self.norm == 'L2':
                        dist1 = df.abs() / (1e-12 + (dg ** 2)
                                            .view(dg.shape[0], dg.shape[1], -1)
                                            .sum(dim=-1).sqrt())
                    elif self.norm == 'L1':
                        dist1 = df.abs() / (1e-12 + dg.abs().reshape(
                            [df.shape[0], df.shape[1], -1]).max(dim=2)[0])
                    else:
                        raise ValueError('norm not supported')
                    ind = dist1.min(dim=1)[1]
                    dg2 = dg[u1, ind]
                    b = (- df[u1, ind] + (dg2 * x1).view(x1.shape[0], -1)
                                         .sum(dim=-1))
                    w = dg2.reshape([bs, -1])

                    if self.norm == 'Linf':
                        d3 = projection_linf(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    elif self.norm == 'L2':
                        d3 = projection_l2(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    elif self.norm == 'L1':
                        d3 = projection_l1(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    d1 = torch.reshape(d3[:bs], x1.shape)
                    d2 = torch.reshape(d3[-bs:], x1.shape)
                    if self.norm == 'Linf':
                        a0 = d3.abs().max(dim=1, keepdim=True)[0]\
                            .view(-1, *[1]*self.ndims)
                    elif self.norm == 'L2':
                        a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()\
                            .view(-1, *[1]*self.ndims)
                    elif self.norm == 'L1':
                        a0 = d3.abs().sum(dim=1, keepdim=True)\
                            .view(-1, *[1]*self.ndims)
                    a0 = torch.max(a0, 1e-8 * torch.ones(
                        a0.shape).to(self.device))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = torch.min(torch.max(a1 / (a1 + a2),
                                                torch.zeros(a1.shape)
                                                .to(self.device)),
                                      self.alpha_max * torch.ones(a1.shape)
                                      .to(self.device))
                    x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                          (im2 + d2 * self.eta) * alpha).clamp(0.0, 1.0)

                    is_adv = self._get_predicted_label(x1) != la2

                    if is_adv.sum() > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        ind_adv = self.check_shape(ind_adv)
                        if self.norm == 'Linf':
                            t = (x1[ind_adv] - im2[ind_adv]).reshape(
                                [ind_adv.shape[0], -1]).abs().max(dim=1)[0]
                        elif self.norm == 'L2':
                            t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                                .view(ind_adv.shape[0], -1).sum(dim=-1).sqrt()
                        elif self.norm == 'L1':
                            t = (x1[ind_adv] - im2[ind_adv])\
                                .abs().view(ind_adv.shape[0], -1).sum(dim=-1)
                        adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                            float().reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                            * (t >= res2[ind_adv]).float().reshape(
                            [-1, *[1]*self.ndims])
                        res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        x1[ind_adv] = im2[ind_adv] + (
                            x1[ind_adv] - im2[ind_adv]) * self.beta

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(ind_succ.float().sum(), corr_classified) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def attack_single_run_targeted(self, x, y=None, use_rand_start=False):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)
        #assert next(self.predict.parameters()).device == x.device

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        corr_classified = pred.float().sum()
        if self.verbose:
            print('Clean accuracy: {:.2%}'.format(pred.float().mean()))
        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        output = self.predict(x)
        la_target = output.sort(dim=-1)[1][:, -self.target_class]

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        la_target2 = la_target[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                if self.norm == 'Linf':
                    t = 2 * torch.rand(x1.shape).to(self.device) - 1
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / (t.reshape([t.shape[0], -1]).abs()
                                         .max(dim=1, keepdim=True)[0]
                                         .reshape([-1, *[1]*self.ndims])) * .5
                elif self.norm == 'L2':
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / ((t ** 2)
                                         .view(t.shape[0], -1)
                                         .sum(dim=-1)
                                         .sqrt()
                                         .view(t.shape[0], *[1]*self.ndims)) * .5
                elif self.norm == 'L1':
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = im2 + (torch.min(res2,
                                          self.eps * torch.ones(res2.shape)
                                          .to(self.device)
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / (t.abs().view(t.shape[0], -1)
                                         .sum(dim=-1)
                                         .view(t.shape[0], *[1]*self.ndims)) / 2

                x1 = x1.clamp(0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.n_iter:
                with torch.no_grad():
                    df, dg = self.get_diff_logits_grads_batch_targeted(x1, la2, la_target2)
                    if self.norm == 'Linf':
                        dist1 = df.abs() / (1e-12 +
                                            dg.abs()
                                            .view(dg.shape[0], dg.shape[1], -1)
                                            .sum(dim=-1))
                    elif self.norm == 'L2':
                        dist1 = df.abs() / (1e-12 + (dg ** 2)
                                            .view(dg.shape[0], dg.shape[1], -1)
                                            .sum(dim=-1).sqrt())
                    elif self.norm == 'L1':
                        dist1 = df.abs() / (1e-12 + dg.abs().reshape(
                            [df.shape[0], df.shape[1], -1]).max(dim=2)[0])
                    else:
                        raise ValueError('norm not supported')
                    ind = dist1.min(dim=1)[1]
                    #print(ind)
                    dg2 = dg[u1, ind]
                    b = (- df[u1, ind] + (dg2 * x1).view(x1.shape[0], -1)
                                         .sum(dim=-1))
                    w = dg2.reshape([bs, -1])

                    if self.norm == 'Linf':
                        d3 = projection_linf(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    elif self.norm == 'L2':
                        d3 = projection_l2(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    elif self.norm == 'L1':
                        d3 = projection_l1(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0))
                    d1 = torch.reshape(d3[:bs], x1.shape)
                    d2 = torch.reshape(d3[-bs:], x1.shape)
                    if self.norm == 'Linf':
                        a0 = d3.abs().max(dim=1, keepdim=True)[0]\
                            .view(-1, *[1]*self.ndims)
                    elif self.norm == 'L2':
                        a0 = (d3 ** 2).sum(dim=1, keepdim=True).sqrt()\
                            .view(-1, *[1]*self.ndims)
                    elif self.norm == 'L1':
                        a0 = d3.abs().sum(dim=1, keepdim=True)\
                            .view(-1, *[1]*self.ndims)
                    a0 = torch.max(a0, 1e-8 * torch.ones(
                        a0.shape).to(self.device))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = torch.min(torch.max(a1 / (a1 + a2),
                                                torch.zeros(a1.shape)
                                                .to(self.device)),
                                      self.alpha_max * torch.ones(a1.shape)
                                      .to(self.device))
                    x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                          (im2 + d2 * self.eta) * alpha).clamp(0.0, 1.0)

                    is_adv = self._get_predicted_label(x1) != la2

                    if is_adv.sum() > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        ind_adv = self.check_shape(ind_adv)
                        if self.norm == 'Linf':
                            t = (x1[ind_adv] - im2[ind_adv]).reshape(
                                [ind_adv.shape[0], -1]).abs().max(dim=1)[0]
                        elif self.norm == 'L2':
                            t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                                .view(ind_adv.shape[0], -1).sum(dim=-1).sqrt()
                        elif self.norm == 'L1':
                            t = (x1[ind_adv] - im2[ind_adv])\
                                .abs().view(ind_adv.shape[0], -1).sum(dim=-1)
                        adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                            float().reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                            * (t >= res2[ind_adv]).float().reshape(
                            [-1, *[1]*self.ndims])
                        res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        x1[ind_adv] = im2[ind_adv] + (
                            x1[ind_adv] - im2[ind_adv]) * self.beta

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(ind_succ.float().sum(), corr_classified) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def perturb(self, x, y):
        adv = x.clone()
        with torch.no_grad():
            acc = self.predict(x).max(1)[1] == y

            startt = time.time()

            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            if not self.targeted:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        adv_curr = self.attack_single_run(x_to_fool, y_to_fool, use_rand_start=(counter > 0))

                        acc_curr = self.predict(adv_curr).max(1)[1] == y_to_fool
                        if self.norm == 'Linf':
                            res = (x_to_fool - adv_curr).abs().view(x_to_fool.shape[0], -1).max(1)[0]
                        elif self.norm == 'L2':
                            res = ((x_to_fool - adv_curr) ** 2).view(x_to_fool.shape[0], -1).sum(dim=-1).sqrt()
                        acc_curr = torch.max(acc_curr, res > self.eps)

                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, acc.float().mean(), self.eps, time.time() - startt))

            else:
                for target_class in range(2, self.n_target_classes + 2):
                    self.target_class = target_class
                    for counter in range(self.n_restarts):
                        ind_to_fool = acc.nonzero().squeeze()
                        if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
                        if ind_to_fool.numel() != 0:
                            x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                            adv_curr = self.attack_single_run_targeted(x_to_fool, y_to_fool, use_rand_start=(counter > 0))

                            acc_curr = self.predict(adv_curr).max(1)[1] == y_to_fool
                            if self.norm == 'Linf':
                                res = (x_to_fool - adv_curr).abs().view(x_to_fool.shape[0], -1).max(1)[0]
                            elif self.norm == 'L2':
                                res = ((x_to_fool - adv_curr) ** 2).view(x_to_fool.shape[0], -1).sum(dim=-1).sqrt()
                            acc_curr = torch.max(acc_curr, res > self.eps)

                            ind_curr = (acc_curr == 0).nonzero().squeeze()
                            acc[ind_to_fool[ind_curr]] = 0
                            adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                            if self.verbose:
                                print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                    counter, self.target_class, acc.float().mean(), self.eps, time.time() - startt))

        return adv

