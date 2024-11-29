import torch
import numpy as np

def get_stats(parameters, deterministic):
    
    mean, logvar = torch.chunk(parameters, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    std = torch.exp(0.5 * logvar)
    var = torch.exp(logvar)
    if deterministic:
        var = std = torch.zeros_like(mean, device=parameters.device, dtype=parameters.dtype)
    return mean, logvar, std, var

def kl_(deterministic, other, mean, var, logvar):
    if deterministic:
        return torch.Tensor([0.])
    else:
        if other is None:
            return torch.sum(torch.pow(mean, 2)
                                    + var - 1.0 - logvar,
                                    dim=list(range(mean.ndim))[1:])
        else:
            return torch.sum(
                torch.pow(mean - other.mean, 2) / other.var
                + var / other.var - 1.0 - logvar + other.logvar,
                dim=list(mean.shape)[1:])

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, parameters_t, deterministic=False):
        self.parameters = parameters
        self.parameters_t = parameters_t.transpose(1, 2) if parameters_t is not None else None
        self.deterministic = deterministic

        if self.parameters is not None:
            self.mean, self.logvar, self.std, self.var = get_stats(self.parameters, self.deterministic)
        if self.parameters_t is not None:
            self.mean_t, self.logvar_t, self.std_t, self.var_t = get_stats(self.parameters_t, self.deterministic)

    def sample(self):
        x, x_t = None, None
        if self.parameters is not None:
            x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device, dtype=self.parameters.dtype)
        if self.parameters_t is not None:
            x_t = self.mean_t + self.std_t * torch.randn(self.mean_t.shape, device=self.parameters_t.device, dtype=self.parameters_t.dtype)
        return x, x_t.transpose(1, 2) if x_t is not None else x_t

    def kl(self, other=None):
        assert other is None
        kl_1, kl_2 = 0.0, 0.0
        if self.parameters is not None:
            kl_1 = kl_(self.deterministic, other, self.mean, self.var, self.logvar)
        if self.parameters_t is not None:
            kl_2 = kl_(self.deterministic, other, self.mean_t, self.var_t, self.logvar_t)
        return kl_1 + kl_2
                
    def mode(self):
        return self.mean, self.mean_t
