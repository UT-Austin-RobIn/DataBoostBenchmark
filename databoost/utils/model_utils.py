import math

import numpy as np
import torch


class Gaussian:
    def __init__(self, mu, log_sigma=None):
        if log_sigma is None:
            mu, log_sigma = torch.chunk(mu, 2, -1)
        self.mu = mu
        self.log_sigma = torch.clamp(log_sigma, min=-10, max=2) \
                         if isinstance(log_sigma, torch.Tensor) else \
                         np.clip(log_sigma, a_min=-10, a_max=2)
        self._sigma = None
        
    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.sigma)

    def kl_divergence(self, other):
        """Here self=q and other=p and we compute KL(q, p)"""
        return (other.log_sigma - self.log_sigma) + (self.sigma ** 2 + (self.mu - other.mu) ** 2) \
               / (2 * other.sigma ** 2) - 0.5

    def nll(self, x):
        return -1 * self.log_prob(x)

    def log_prob(self, val):
        return -1 * ((val - self.mu) ** 2) / (2 * self.sigma**2) - self.log_sigma - math.log(math.sqrt(2*math.pi))

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.sigma)

    @property
    def sigma(self):
        if self._sigma is None:
            self._sigma = self.log_sigma.exp()
        return self._sigma

    @property
    def shape(self):
        return self.mu.shape

    @staticmethod
    def stack(*argv, dim):
        return Gaussian._combine(torch.stack, *argv, dim=dim)

    @staticmethod
    def cat(*argv, dim):
        return Gaussian._combine(torch.cat, *argv, dim=dim)

    @staticmethod
    def _combine(fcn, *argv, dim):
        mu, log_sigma = [], []
        for g in argv:
            mu.append(g.mu)
            log_sigma.append(g.log_sigma)
        mu = fcn(mu, dim)
        log_sigma = fcn(log_sigma, dim)
        return Gaussian(mu, log_sigma)

    def average(self, dists):
        """Fits single Gaussian to a list of Gaussians."""
        mu_avg = torch.stack([d.mu for d in dists]).sum(0) / len(dists)
        sigma_avg = torch.stack([d.mu ** 2 + d.sigma ** 2 for d in dists]).sum(0) - mu_avg**2
        return type(self)(mu_avg, torch.log(sigma_avg))

    def chunk(self, *args, **kwargs):
        return [type(self)(chunk) for chunk in torch.chunk(self.tensor(), *args, **kwargs)]

    def view(self, shape):
        self.mu = self.mu.view(shape)
        self.log_sigma = self.log_sigma.view(shape)
        self._sigma = self.sigma.view(shape)
        return self

    def __getitem__(self, item):
        return Gaussian(self.mu[item], self.log_sigma[item])
 
    def tensor(self):
        return torch.cat([self.mu, self.log_sigma], dim=-1)

    def rsample(self):
        """Identical to self.sample(), to conform with pytorch naming scheme."""
        return self.sample()

    def detach(self):
        """Detaches internal variables. Returns detached Gaussian."""
        return type(self)(self.mu.detach(), self.log_sigma.detach())

    def to_numpy(self):
        """Convert internal variables to numpy arrays."""
        return type(self)(ten2ar(self.mu), ten2ar(self.log_sigma))


class MultivariateGaussian(Gaussian):
    def log_prob(self, val):
        return super().log_prob(val).sum(-1)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.sigma.sum(-1))

    @staticmethod
    def stack(*argv, dim):
        return MultivariateGaussian(Gaussian.stack(*argv, dim=dim).tensor())

    @staticmethod
    def cat(*argv, dim):
        return MultivariateGaussian(Gaussian.cat(*argv, dim=dim).tensor())


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)