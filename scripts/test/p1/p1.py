from utils import to_variable
import pyro
import pyro.distributions as dist
import torch
from pdb import set_trace as bb
def init_params(params):
    params["mu"] = 0.
    params["sigma"] = 1.

def model(data, params):
    N = data["N"]
    y = to_variable(data["y"])
    mu = to_variable(params["mu"])
    sigma = to_variable(params["sigma"])

    pyro.sample("y", dist.Normal(mu,sigma), obs=y)