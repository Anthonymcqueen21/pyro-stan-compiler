from utils import to_variable
import pyro
import pyro.distributions as dist
import torch

def init_params(params):
    params["mu"] = 0.
    params["sigma"] = 1.

EPSILON = 1e-6
def model(data, params):
    #N = data["N"]
    y = to_variable(data["y"])
    #mu = to_variable(params["mu"])
    #sigma = to_variable(params["sigma"])
    TWO = to_variable(2.)

    mu = pyro.sample("mu", dist.Uniform(-TWO,TWO))
    log_sigma = pyro.sample("log_sigma", dist.Uniform(-TWO, TWO))
    sigma = torch.exp(log_sigma)
    if sigma < EPSILON:
        sigma = EPSILON
    #xm = pyro.sample("xm", dist.Normal(mu,sigma))
    pyro.sample("y", dist.Normal(mu,sigma), obs=y)