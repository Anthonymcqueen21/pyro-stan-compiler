import torch
from torch.autograd import Variable
import pyro
import pyro.distributions as dist
from pyro.util import ng_ones, ng_zeros
import pandas

# TODO: load data from csv
x = Variable(torch.Tensor(
    [
      [151,199,246,283,320],
      [145,199,249,293,354],
      [147,214,263,312,328],
      [155,200,237,272,297],
      [135,188,230,280,323],
      [159,210,252,298,331],
      [141,189,231,275,305],
      [159,201,248,297,338],
      [177,236,285,350,376],
      [134,182,220,260,296],
      [160,208,261,313,352],
      [143,188,220,273,314],
      [154,200,244,289,325],
      [171,221,270,326,358],
      [163,216,242,281,312],
      [160,207,248,288,324],
      [142,187,234,280,316],
      [156,203,243,283,317],
      [157,212,259,307,336],
      [152,203,246,286,321],
      [154,205,253,298,334],
      [139,190,225,267,302],
      [146,191,229,272,302],
      [157,211,250,285,323],
      [132,185,237,286,331],
      [160,207,257,303,345],
      [169,216,261,295,333],
      [157,205,248,289,316],
      [137,180,219,258,291],
      [153,200,244,286,324]
    ]
))

# init
N = 30
T = 5
xbar = 22

# ng(value, dim0, dim1)
def ng(*args, **kwargs):
    if len(args) < 3:
	raise ValueError("args msut be = 3")
    return Variable(torch.Tensor(args[1], args[2]).fill_(args[0]))

def model(x_minus_xbar):
    # sampled params
    mu_alpha = pyro.sample("mu_alpha", dist.normal, ng_zeros(N, 1), ng(100, N, 1))
    mu_beta = pyro.sample("mu_beta", dist.normal, ng_zeros(N, 1), ng(100, N, 1))
    sigmasq_y = pyro.sample("sigmasq_y", dist.halfcauchy, ng(1e-3, N, T), ng(1e-3, N, T))
    sigmasq_alpha = pyro.sample("sigmasq_alpha", dist.halfcauchy, ng(1e-3, N, 1), ng(1e-3, N, 1))
    sigmasq_beta = pyro.sample("sigmasq_beta", dist.halfcauchy, ng(1e-3, N, 1), ng(1e-3, N, 1))

    # transformed parameters
    sigma_y = torch.sqrt(sigmasq_y)
    sigma_alpha = torch.sqrt(sigmasq_alpha)
    sigma_beta = torch.sqrt(sigmasq_beta)

    alpha = pyro.sample("alpha", dist.halfcauchy, mu_alpha, sigma_alpha)
    beta = pyro.sample("beta", dist.halfcauchy, mu_beta, sigma_beta)
    # observe
    pred = torch.addmm(beta.expand(N, T), alpha, x_minus_xbar.t())
    pyro.sample("obs", dist.normal, pred, sigma_y, obs=x)

# generate guide from model
def guide(x):
    pass

# TODO: training loop
# transformed data
x_minus_xbar = x - xbar
model(x_minus_xbar[0].unsqueeze(1))
