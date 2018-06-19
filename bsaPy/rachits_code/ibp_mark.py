"""
This file implements completely a mean-field IBP model, including training and evaluation methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import numpy as np
import pdb  # noqa: F401
from collections import OrderedDict

from utils import SMALL, log_sum_exp  # noqa: F401
import shared


class MF_IBP(nn.Module):

    def __init(self,K=10,a=1,b=1,sigma_n=0.5, sigma_):
        super(MF_IBP, self).__init__()
        self.truncation = K
        self.num_features = K
        self.sigma_n = sigma_n
        self.N = 100

        # phi are the logs of the frequencies of the sines
        self.phi_mean = nn.Parameter(torch.zeros(self.num_features,1))
        self.phi_logvar = nn.Parameter(torch.zeros(self.num_features,1))

        # weights
        self.w_mean = nn.Parameter(torch.zeros(self.N, self.num_features))
        self.w_logvar = nn.Parameter(torch.zeros(self.N, self.num_features))

        self.p_pi_alpha = a/float(K)
        self.p_pi_beta = b*(K-1)/float(K)

        # inverse softplus
        a_val = np.log(np.exp(self.p_pi_alpha) - 1)
        b_val = np.log(np.exp(self.p_pi_beta) - 1)
        self.q_pi_alpha = nn.Parameter(torch.Tensor(self.truncation).zero_() + a_val)
        self.q_pi_beta = nn.Parameter(torch.Tensor(self.truncation).zero_() + b_val)
        
        # These are broadcast up into the right shape (they are diagonal)
        self.p_phi = distributions.Normal(loc=mu_phi,scale=sigma_phi)
        self.p_w = distributions.Normal(loc=mu_w,scale=sigma_w)


    def forward(self, x):

        batch_sz = x.size()[0]
        sz = self.q_pi_a.size()
        
        p_pi = distributions.Beta(torch.ones(sz)*self.p_pi_alpha,torch.ones(sz)*self.p_pi_beta)

        beta_a = F.softplus(self.q_pi_alpha) + 0.01
        beta_b = F.softplus(self.q_pi_beta) + 0.01
        q_pi = distributions.Beta(beta_a, beta_b)

       # Differentiable Sample Knowles et al. 
        qpi_sample = q_pi.rsample() 
        q_z = shared.STRelaxedBernoulli(temperature=0.1,probs=qpi_sample)
        z = q_z.rsample() 
        q_z = distributions.Bernoulli(probs=qpi_sample)
        
        q_phi = distributions.Normal(loc=self.phi_mean,scale=(self.phi_logvar/2).exp())
        q_w = distributions.Normal(loc=self.w_mean,scale=(self.w_logvar/2).exp())
        
        # For now, just take the mean
        phi = q_phi.mean
        w = q_w.mean

        # Alternatively, sample
        # phi = q_phi.rsample()
        # w = q_w.rsample()

        # NLL
        sinbasis = torch.ones(K,N_SAMPLES)*torch.arange(0,N_SAMPLES,1)

        for k in range(K):
            sinbasis[k] = torch.sin(sinbasis[k]*phi[k])
        
        x_mean = torch.mm(torch.mul(z,w),sinbasis) # z and w multiplied elementwise
        nll = -(distributions.Normal(loc=x_mean, scale=self.sigma_n).log_prob(x))
        return nll, p_pi, q_pi, q_z, q_phi, q_w, sinbasis

    



















    def elbo(nll, p_pi, q_pi, p_z, q_z, p_a, q_a, batch_sz, sz):
        kl_divergence = distributions.kl_divergence
        components = (
            nll.sum(1),
            kl_divergence(q_pi, p_pi).sum().repeat(batch_sz) / sz,
            kl_divergence(q_z, p_z).sum(1),
            kl_divergence(q_a, p_a).sum().repeat(batch_sz) / sz
        )
        return components

    @staticmethod
    def iwae(nll, p_pi, q_pi, p_z, q_z, p_a, q_a, batch_sz, sz, num_importance_samples):
        kl_divergence = distributions.kl_divergence

        # the global variables are repeated (because sampled once per batch - local are not scaled
        logK = np.log(num_importance_samples)
        components = (
            -log_sum_exp(-nll.sum(1).view(num_importance_samples, batch_sz), 0) + logK,
            kl_divergence(q_pi, p_pi).sum().repeat(batch_sz) / sz,
            -log_sum_exp(-kl_divergence(q_z, p_z).sum(1).view(num_importance_samples, batch_sz), 0) + logK,
            kl_divergence(q_a, p_a).sum().repeat(batch_sz) / sz
        )
        return components

    def compute_train_loss(self, x, sz, iwae, n_samples=1):
        """
        During training, this function does the heavy lifting:
        1. compute the distributions required for VI
        2. compute the ELBO (here, -loss)
        3. find the REINFORCE loss (the derivatives of which are the REINFORCE gradients)
        4. return (something to optimize), ELBO
        """
        batch_sz = x.size(0)
        if iwae:
            x = x.repeat(n_samples, 1)
        nll, p_pi, q_pi, p_z, q_z, p_a, q_a = self.forward(x)
        if iwae:
            loss = sum(self.iwae(nll, p_pi, q_pi, p_z, q_z, p_a, q_a, batch_sz, sz, n_samples))
        else:
            loss = sum(self.elbo(nll, p_pi, q_pi, p_z, q_z, p_a, q_a, batch_sz, sz))
        # reinforce_loss = self.z_log_prob.sum(1) * loss.data  # NOTE `.data` so that this doesn't backprop to generative model
        return loss.sum(), -loss.sum()


    def compute_eval_loss(self, x, sz, num_importance_samples):
        """
        Computes the evaluation-level loss - we want finer grained logging at this level
        """
        batch_sz = x.size(0)
        x = x.repeat(num_importance_samples, 1)
        nll, p_pi, q_pi, p_z, q_z, p_a, q_a = self.forward(x)
        loss = self.iwae(nll, p_pi, q_pi, p_z, q_z, p_a, q_a, batch_sz, sz, num_importance_samples)
        return map(lambda x: x.sum().item(), loss)



    def train_epoch(self, train_loader, optimizer, epoch, args, device, n_samples):
        self.train()
        sz = len(train_loader.dataset)
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device).float()
            loss, elbo = self.compute_train_loss(batch, sz, args.train_iwae, n_samples)
            loss.backward()
            optimizer.step()
            total_loss += elbo.item()

            if (batch_idx + 1) % args.log_interval == 0:
                pass

        return total_loss / (sz * n_samples)





    '''
    def evaluate(self, dataset, args, device, num_importance_samples):
        """
        Returns `ret`, which contains summary statistics about the evaluation on the `dataset`
        """
        self.eval()
        sz = len(dataset.dataset)
        total_nll = 0
        total_kl_pi = 0
        total_kl_z = 0
        total_kl_a = 0

        for batch_idx, batch in enumerate(dataset):
            batch = batch.to(device).float()
            nll, kl_pi, kl_z, kl_a = self.compute_eval_loss(batch, sz, num_importance_samples)
            total_nll += nll
            total_kl_pi += kl_pi
            total_kl_z += kl_z
            total_kl_a += kl_a

        ret = OrderedDict([
            ('ELBO', -(total_nll + total_kl_pi + total_kl_z + total_kl_a) / sz),
            ('NLL', total_nll / sz),
            ('KL_pi', total_kl_pi),
            ('KL_Z', total_kl_z / sz),
            ('KL_A', total_kl_a),
        ])
        return ret
    ''' 




