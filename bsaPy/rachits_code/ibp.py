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


class MF_IBP_VAE(nn.Module):
    """
    This is a mean-field, BBVI version of the IBP-VAE - for comparison against the paintbox models.
    """

    def __init__(self, max_truncation_level=6, alpha0=5., sigma_n=0.5, D=36, init=None):
        super(MF_IBP_VAE, self).__init__()

        self.truncation = max_truncation_level
        self.num_features = max_truncation_level
        self.D = D
        self.sigma_n = sigma_n

        # inference network: image to logits, one upweighting for each feature (i.e. a recognition net)
        self.encoder = nn.Linear(self.D, self.num_features)

        # generator network: torch.mm(Z, A)
        self.A_mean = nn.Parameter(torch.zeros(self.num_features, self.D))
        self.A_logvar = nn.Parameter(torch.zeros(self.num_features, self.D))

        a_val = np.log(np.exp(alpha0) - 1)  # inverse softplus
        b_val = np.log(np.exp(1.) - 1)

        # parameters for the inference tree - note that here there's one for every (k, j) - there's a mapper inside
        self.beta_a = nn.Parameter(torch.Tensor(self.truncation).zero_() + a_val)
        self.beta_b = nn.Parameter(torch.Tensor(self.truncation).zero_() + b_val)
        self.alpha0 = alpha0

        # initialize
        torch.nn.init.normal_(self.encoder.weight.data, 0, 0.01)
        self.encoder.bias.data.zero_()

        if init is not None:
            self.A_mean.data = init.float()
        # self.A_logvar.data -= 10
        # self.A_logvar.requires_grad = False

    #
    # Core Logic
    #
    def forward(self, x):
        """
        This function takes a batch of data `x` and returns:
        - nll: -\E_q[log p(x | z, A)]
        - q_z: a torch Bernoulli Distribution for q(z)
        - p_z: a torch Bernoulli Distribution for p(z | nu) *where nu ~ q(nu)* [because it's for the KL divergence]
        - q_nu: a torch Beta Distribution for q(nu)
        - p_nu: a torch Beta Distribution for p(nu)
        - q_a: a torch Normal Distribution (univariate / diagonal) for q(A)
        - p_a: a torch Normal Distribution for p(A)

        The negative ELBO can be computed as:
        -ELBO = nll + KL(q_z || p_z) + KL(q_nu || p_nu)
        """
        batch_sz = x.size()[0]

        # p(nu)
        sz = self.beta_a.size()
        p_nu = distributions.Beta(torch.ones(sz) * self.alpha0, torch.ones(sz))

        # compute q(nu) parameters, and take samples
        beta_a = F.softplus(self.beta_a) + 0.01
        beta_b = F.softplus(self.beta_b) + 0.01
        q_nu = distributions.Beta(beta_a, beta_b)

        nu = q_nu.rsample()  # NOTE: differentiable sample! via Knowles et al.

        # p(z | nu)
        logpi = torch.cumsum((nu + SMALL).log(), dim=-1).unsqueeze(0).repeat(batch_sz, 1)
        p_z = distributions.Bernoulli(probs=logpi.exp())

        # q(z)
        # machine/fp precision is higher near 0 than at 1 (crucial)
        probs = F.sigmoid(torch.clamp(self.encoder(x.view(-1, self.D)), -25, 9))
        q_z = shared.STRelaxedBernoulli(temperature=0.1, probs=probs)
        # q_z = distributions.RelaxedBernoulli(temperature=0.2, probs=probs)
        z = q_z.rsample()
        q_z = distributions.Bernoulli(probs=probs)
        # self.z_log_prob = q_z.log_prob(z)  # save for later

        # p(A)
        p_a = distributions.Normal(loc=0, scale=1)  # NOTE: this is broadcast up

        # q(A) - this is wrong, it normalizes the wrong thing
        q_a = distributions.Normal(loc=self.A_mean, scale=(self.A_logvar/2).exp())

        A = self.A_mean
        # A = q_a.rsample()

        # now compute NLL:
        x_mean = torch.mm(z, A)
        nll = -(distributions.Normal(loc=x_mean, scale=self.sigma_n).log_prob(x))

        return nll, p_nu, q_nu, p_z, q_z, p_a, q_a

    @staticmethod
    def elbo(nll, p_nu, q_nu, p_z, q_z, p_a, q_a, batch_sz, sz):
        kl_divergence = distributions.kl_divergence
        components = (
            nll.sum(1),
            kl_divergence(q_nu, p_nu).sum().repeat(batch_sz) / sz,
            kl_divergence(q_z, p_z).sum(1),
            kl_divergence(q_a, p_a).sum().repeat(batch_sz) / sz
        )
        return components

    @staticmethod
    def iwae(nll, p_nu, q_nu, p_z, q_z, p_a, q_a, batch_sz, sz, num_importance_samples):
        kl_divergence = distributions.kl_divergence

        # the global variables are repeated (because sampled once per batch - local are not scaled
        logK = np.log(num_importance_samples)
        components = (
            -log_sum_exp(-nll.sum(1).view(num_importance_samples, batch_sz), 0) + logK,
            kl_divergence(q_nu, p_nu).sum().repeat(batch_sz) / sz,
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
        nll, p_nu, q_nu, p_z, q_z, p_a, q_a = self.forward(x)
        if iwae:
            loss = sum(self.iwae(nll, p_nu, q_nu, p_z, q_z, p_a, q_a, batch_sz, sz, n_samples))
        else:
            loss = sum(self.elbo(nll, p_nu, q_nu, p_z, q_z, p_a, q_a, batch_sz, sz))
        # reinforce_loss = self.z_log_prob.sum(1) * loss.data  # NOTE `.data` so that this doesn't backprop to generative model
        return loss.sum(), -loss.sum()

    def compute_eval_loss(self, x, sz, num_importance_samples):
        """
        Computes the evaluation-level loss - we want finer grained logging at this level
        """
        batch_sz = x.size(0)
        x = x.repeat(num_importance_samples, 1)
        nll, p_nu, q_nu, p_z, q_z, p_a, q_a = self.forward(x)
        loss = self.iwae(nll, p_nu, q_nu, p_z, q_z, p_a, q_a, batch_sz, sz, num_importance_samples)
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

    def evaluate(self, dataset, args, device, num_importance_samples):
        """
        Returns `ret`, which contains summary statistics about the evaluation on the `dataset`
        """
        self.eval()
        sz = len(dataset.dataset)
        total_nll = 0
        total_kl_nu = 0
        total_kl_z = 0
        total_kl_a = 0

        for batch_idx, batch in enumerate(dataset):
            batch = batch.to(device).float()
            nll, kl_nu, kl_z, kl_a = self.compute_eval_loss(batch, sz, num_importance_samples)
            total_nll += nll
            total_kl_nu += kl_nu
            total_kl_z += kl_z
            total_kl_a += kl_a

        ret = OrderedDict([
            ('ELBO', -(total_nll + total_kl_nu + total_kl_z + total_kl_a) / sz),
            ('NLL', total_nll / sz),
            ('KL_nu', total_kl_nu),
            ('KL_Z', total_kl_z / sz),
            ('KL_A', total_kl_a),
        ])
        return ret
