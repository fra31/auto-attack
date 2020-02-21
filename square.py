import numpy as np
import torch
import time

class ModelAdapterSA():
    def __init__(self, model):
        self.model = model
    
    def predict(self, x):
        return self.model(x)

    def fmargin(self, x, y):
        logits = self.predict(x)
        u = torch.arange(x.shape[0])
        y_corr = logits[u, y]
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]
        
        return y_corr - y_others

class SquareAttack():
    def __init__(self, model, norm='Linf', n_queries=5000, eps=.3, p_init=.8, device='cuda',
                 early_stop=True, eot_iter=1, n_restarts=1, seed=0, verbose=False):
        self.model = ModelAdapterSA(model)
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.device = device
        self.early_stop = early_stop
        self.eot_iter = eot_iter
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
    
    def check_shape(self, x):
        return x if len(x.shape) == 4 else x.unsqueeze(0)
    
    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = torch.zeros([x, y])
        x_c, y_c = x//2 + 1, y//2 + 1
    
        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
          delta[max(counter2[0], 0):min(counter2[0] + (2*counter + 1), x), max(0, counter2[1]):min(counter2[1] + (2*counter + 1), y)] += 1.0/(torch.Tensor([counter + 1]).reshape([1, 1]) ** 2)
          counter2[0] -= 1
          counter2[1] -= 1
        
        delta /= (delta ** 2).sum(dim=(0,1), keepdim=True).sqrt()
    
        return delta.to(self.device)

    def meta_pseudo_gaussian_pert(self, s):
        delta = torch.zeros([s, s])
        delta[:s//2] = self.pseudo_gaussian_pert_rectangles(s//2, s)
        delta[s//2:] = self.pseudo_gaussian_pert_rectangles(s - s//2, s)*(-1)
        delta /= (delta ** 2).sum(dim=(0,1), keepdim=True).sqrt()
        if torch.rand([1]) > 0.5: delta = delta.permute([1, 0])
        
        return delta.to(self.device)

    def p_selection(self, it):
        #it = int(it / self.n_queries * 10000)
        
        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init
        
        return p

    def attack_single_run(self, x_in, y_in):
        with torch.no_grad():
            x, y = x_in.clone(), y_in.clone()
            
            if self.norm == 'Linf':
                if self.eot_iter == 1:
                    output = self.model.predict(x)
                    corr_classified = (output.max(dim=1)[1] == y).nonzero().squeeze()
                else:
                    output = self.model.predict(x)
                    corr_classified = (output.max(dim=1)[1] == y).float() / self.eot_iter
                    for _ in range(self.eot_iter - 1):
                        output = self.model.predict(x)
                        corr_classified += ((output.max(dim=1)[1] == y).float() / self.eot_iter)
                    
                    corr_classified = (corr_classified > .5).nonzero().squeeze()
                
                adv = x.clone()
                n_queries_complete = torch.zeros(x.shape[0]).to(self.device)
                corr_cl_init = corr_classified.clone()
                
                c, h, w = x.shape[1:]
                n_features = c * h * w
                n_ex_total = x.shape[0]
                x, y = x.clone()[corr_classified], y.clone()[corr_classified]
                x = self.check_shape(x)
                if len(y.shape) == 0:
                    y = y.unsqueeze(0)
                
                x_best = torch.clamp(x + torch.from_numpy(self.eps * np.random.choice([-1, 1], size=[x.shape[0], c, 1, w])).float().to(self.device), 0., 1.)

                if self.eot_iter == 1:
                    margin_min = self.model.fmargin(x_best, y)
                else:
                    margin_min = torch.zeros(x.shape[0]).to(self.device)
                    for _ in range(self.eot_iter):
                        margin_min_temp = self.model.fmargin(x_best, y)
                        margin_min += (margin_min_temp / self.eot_iter)
                    
                n_queries = torch.ones(x.shape[0]).to(self.device)
                
                s_init = int(np.sqrt(self.p_init * n_features / c))
                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().squeeze() if self.early_stop else (margin_min > -float('inf')).nonzero().squeeze()
                    
                    x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
                    y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
            
                    p = self.p_selection(i_iter)
                    s = max(int(round(np.sqrt(p * n_features / c))), 1)
                    center_h = np.random.randint(0, h-s)
                    center_w = np.random.randint(0, w-s)
                    new_deltas = torch.zeros([c, h, w]).to(self.device)
                    new_deltas[:, center_h:center_h+s, center_w:center_w+s] = torch.from_numpy(np.random.choice([-2*self.eps, 2*self.eps], size=[c, 1, 1])).to(self.device)
                    
                    x_new = x_best_curr + new_deltas
                    x_new = torch.min(torch.max(x_new, x_curr - self.eps * torch.ones(x_curr.shape).to(self.device)), x_curr + self.eps * torch.ones(x_curr.shape).to(self.device))
                    x_new = torch.clamp(x_new, 0., 1.)
        
                    x_new = self.check_shape(x_new)
                    if self.eot_iter == 1:
                        margin = self.model.fmargin(x_new, y_curr)
                    else:
                        margin = torch.zeros(x_new.shape[0]).to(self.device)
                        for _ in range(self.eot_iter):
                            margin_min_temp = self.model.fmargin(x_new, y_curr)
                            margin += (margin_min_temp / self.eot_iter)
                    
                    idx_improved = (margin < margin_min_curr).float()
                    margin_min[idx_to_fool] = idx_improved * margin + (1. - idx_improved) * margin_min_curr
                    idx_improved = idx_improved.reshape([-1, *[1]*len(x.shape[:-1])])
                    x_best[idx_to_fool] = idx_improved * x_new + (1. - idx_improved) * x_best_curr
                    n_queries[idx_to_fool] += 1.
            
                    acc = (margin_min > 0.0).sum().float() / n_ex_total
                    acc_corr = (margin_min > 0.0).float().mean()
                    ind_succ = (margin_min <= 0.).nonzero().squeeze()
                    
                        
                    if self.verbose and ind_succ.numel() != 0:
                        print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} loss={:.3f}'.
                            format(i_iter + 1, acc.item(), acc_corr.item(), n_queries[ind_succ].mean().item(), n_queries[ind_succ].median().item(), margin_min.mean()))
                    
                    if acc == 0 and self.verbose:
                        print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} loss={:.3f}'.
                            format(i_iter + 1, acc.item(), acc_corr.item(), n_queries[ind_succ].mean().item(), n_queries[ind_succ].median().item(), margin_min.mean()))
                        break
              
                adv[corr_cl_init] = x_best.clone()
                n_queries_complete[corr_cl_init] = n_queries.clone()
            
            elif self.norm == 'L2':
                output = self.model.predict(x)
                corr_classified = (output.max(dim=1)[1] == y).nonzero().squeeze()
                
                adv = x.clone()
                n_queries_complete = torch.zeros(x.shape[0]).to(self.device)
                corr_cl_init = corr_classified.clone()
                
                c, h, w = x.shape[1:]
                n_features = c * h * w
                n_ex_total = x.shape[0]
                x, y = x.clone()[corr_classified], y.clone()[corr_classified]
                delta_init = torch.zeros(x.shape).to(self.device)
                s = h//5
                sp_init = (h - s*5)//2
                center_h = sp_init + 0
                for counter in range(h//s):
                    center_w = sp_init + 0
                    for counter2 in range(w//s):
                        delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += self.meta_pseudo_gaussian_pert(s).reshape(
                            [1, 1, s, s]) * torch.from_numpy(np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])).float().to(self.device)
                        center_w += s
                    center_h += s
                  
                x_best = torch.clamp(x + delta_init/(delta_init**2).sum(dim=(1,2,3), keepdim=True).sqrt() * self.eps, 0., 1.)
                margin_min = self.model.fmargin(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                s_init = int(np.sqrt(self.p_init * n_features / c))
                
                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().squeeze() if self.early_stop else (margin_min > -float('inf')).nonzero().squeeze()
                    
                    x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
                    y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
            
                    delta_curr = x_best_curr - x_curr
                    p = self.p_selection(i_iter)
                    s = max(int(round(np.sqrt(p * n_features / c))), 3)
                    if s % 2 == 0: s += 1
                    s2 = s + 0
                    
                    center_h = np.random.randint(0, h-s)
                    center_w = np.random.randint(0, w-s)
                    new_deltas_mask = torch.zeros(x_curr.shape).to(self.device)
                    new_deltas_mask[:, :, center_h:center_h+s, center_w:center_w+s] = 1.0
                    norms_window_1 = (delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] ** 2).sum(dim=(-2, -1), keepdim=True).sqrt()
                    
                    center_h_2 = np.random.randint(0, h-s2)
                    center_w_2 = np.random.randint(0, w-s2)
                    new_deltas_mask_2 = torch.zeros(x_curr.shape).to(self.device)
                    new_deltas_mask_2[:, :, center_h_2:center_h_2+s2, center_w_2:center_w_2+s2] = 1.0
                    
                    norms_image = ((x_best_curr - x_curr) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                    mask_image = torch.max(new_deltas_mask, new_deltas_mask_2)
                    norms_windows = ((delta_curr * mask_image) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                    new_deltas = torch.ones([x_curr.shape[0], c, s, s]).to(self.device)
                    new_deltas = new_deltas * self.meta_pseudo_gaussian_pert(s).reshape(
                        [1, 1, s, s]) * torch.from_numpy(np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])).float().to(self.device)
                    old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + norms_window_1)
                    new_deltas += old_deltas
                    new_deltas = new_deltas / (new_deltas ** 2).sum(dim=(-2, -1), keepdim=True).sqrt() * (torch.max(
                        (self.eps * torch.ones(new_deltas.shape).to(self.device)) ** 2 - norms_image ** 2, torch.zeros(new_deltas.shape).to(
                        self.device)) / c + norms_windows ** 2).sqrt()
                    delta_curr[:, :, center_h_2:center_h_2+s2, center_w_2:center_w_2+s2] = 0.0
                    delta_curr[:, :, center_h:center_h+s, center_w:center_w+s] = new_deltas + 0
                        
                    x_new = torch.clamp(x_curr + self.eps * torch.ones(x_curr.shape).to(self.device) * delta_curr/(delta_curr ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt(), 0. ,1.)
                    
                    norms_image = ((x_new - x_curr) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                    
                    margin = self.model.fmargin(x_new, y_curr)
                    idx_improved = (margin < margin_min_curr).float()
                    margin_min[idx_to_fool] = idx_improved * margin + (1. - idx_improved) * margin_min_curr
                    idx_improved = idx_improved.reshape([-1, *[1]*len(x.shape[:-1])])
                    x_best[idx_to_fool] = idx_improved * x_new + (1. - idx_improved) * x_best_curr
                    n_queries[idx_to_fool] += 1.
            
                    acc = (margin_min > 0.0).sum().float() / n_ex_total
                    acc_corr = (margin_min > 0.0).float().mean()
                    ind_succ = (margin_min <= 0.).nonzero().squeeze()
                    if acc_corr < 1. and self.verbose:
                        print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} loss={:.3f} max_pert={:.1f}'.
                            format(i_iter + 1, acc.item(), acc_corr.item(), n_queries[ind_succ].mean().item(), n_queries[ind_succ].median().item(), margin_min.mean(), norms_image.max().item()))
                    
                    if acc == 0:
                        break
              
                adv[corr_cl_init] = x_best.clone()
                n_queries_complete[corr_cl_init] = n_queries.clone() 
            
            return n_queries_complete, adv
            
    def perturb(self, x, y):
        adv = x.clone()
        acc = self.model.predict(x).max(1)[1] == y
        
        startt = time.time()
        
        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        
        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                _, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                if self.eot_iter == 1:
                    acc_curr = self.model.predict(adv_curr).max(1)[1] == y_to_fool
                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                else:
                    output = self.model.predict(adv_curr)
                    corr_classified = (output.max(dim=1)[1] == y_to_fool).float() / self.eot_iter
                    for _ in range(self.eot_iter - 1):
                        output = self.model.predict(adv_curr)
                        corr_classified += ((output.max(dim=1)[1] == y_to_fool).float() / self.eot_iter)
                    
                    ind_curr = (corr_classified < 1.).nonzero().squeeze()
                    print(corr_classified)
                
                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose: print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(counter, acc.float().mean(), time.time() - startt))
            
        return 0, adv