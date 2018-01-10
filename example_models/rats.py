import torch
from torch.autograd import Variable
import pyro
from pyro.infer import SVI
from pyro.optim import Adam
from torch.nn import Softplus
import pyro.distributions as dist
from pyro.util import ng_ones, ng_zeros
import pandas

sp = Softplus()

# TODO: load data from csv
y = Variable(torch.Tensor(
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
x = Variable(torch.Tensor([8.0, 15.0, 22.0, 29.0, 36.0]))
xbar = 22
x_minus_xbar = x - xbar

# ng(value, dim0, dim1)
def ng(*args, **kwargs):
    if len(args) < 3:
	raise ValueError("args msut be = 3")
    return Variable(torch.Tensor(args[1], args[2]).fill_(args[0]))

def model(y):
    # sampled params
    mu_alpha = pyro.sample("mu_alpha", dist.normal, ng_zeros(N, 1), ng(10, N, 1))
    mu_beta = pyro.sample("mu_beta", dist.normal, ng_zeros(N, 1), ng(10, N, 1))
    sigma_y = pyro.sample("sigmasq_y", dist.halfcauchy, ng(1e-3, N, T), ng(1e-3, N, T))
    sigma_alpha = pyro.sample("sigmasq_alpha", dist.halfcauchy, ng(1e-3, N, 1), ng(1e-3, N, 1))
    sigma_beta = pyro.sample("sigmasq_beta", dist.halfcauchy, ng(1e-3, N, 1), ng(1e-3, N, 1))

    # transformed parameters
    # dont need this because pyro dists take std dev not variance
#     sigma_y = torch.sqrt(sigmasq_y)
#     sigma_alpha = torch.sqrt(sigmasq_alpha)
#     sigma_beta = torch.sqrt(sigmasq_beta)

    alpha = pyro.sample("alpha", dist.normal, mu_alpha, sigma_alpha)
    beta = pyro.sample("beta", dist.normal, mu_beta, sigma_beta)
    # observe
    pred = torch.addmm(beta.expand(N, T), alpha, x_minus_xbar.unsqueeze(1).t())
    pyro.sample("obs", dist.normal, pred, sigma_y, obs=y)

# generate guide from model
def guide(x):
    # declare params
    mumu_alpha = pyro.param("mumu_alpha", Variable(torch.randn(N, 1), requires_grad=True))
    sigmamu_alpha = pyro.param("sigmamu_alpha", Variable(torch.rand(N, 1), requires_grad=True))
    mumu_beta = pyro.param("mumu_beta", Variable(torch.randn(N, 1), requires_grad=True))
    sigmamu_beta = pyro.param("sigmamu_beta", Variable(torch.rand(N, 1), requires_grad=True))
    musigma_y = pyro.param("musigma_y", Variable(torch.rand(N, T), requires_grad=True))
    sigmasigma_y = pyro.param("sigmasigma_y", Variable(torch.rand(N, T), requires_grad=True))
    musigma_alpha = pyro.param("musigma_alpha", Variable(torch.rand(N, 1), requires_grad=True))
    sigmasigma_alpha = pyro.param("sigmasigma_alpha", Variable(torch.rand(N, 1), requires_grad=True))
    musigma_beta = pyro.param("musigma_beta", Variable(torch.rand(N, 1), requires_grad=True))
    sigmasigma_beta = pyro.param("sigmasigma_beta", Variable(torch.rand(N, 1), requires_grad=True))

    # sampled params
    mu_alpha = pyro.sample("mu_alpha", dist.normal, mumu_alpha, sp(sigmamu_alpha))
    mu_beta = pyro.sample("mu_beta", dist.normal, mumu_beta, sp(sigmamu_beta))
    sigma_y = pyro.sample("sigmasq_y", dist.halfcauchy, sp(musigma_y), sp(sigmasigma_y))
    sigma_alpha = pyro.sample("sigmasq_alpha", dist.halfcauchy, sp(musigma_alpha), sp(sigmasigma_alpha))
    sigma_beta = pyro.sample("sigmasq_beta", dist.halfcauchy, sp(musigma_beta), sp(sigmasigma_beta))

    # transformed parameters
    # see model() comments
#     sigma_y = torch.sqrt(sigmasq_y)
#     sigma_alpha = torch.sqrt(sigmasq_alpha)
#     sigma_beta = torch.sqrt(sigmasq_beta)

    alpha = pyro.sample("alpha", dist.halfcauchy, mu_alpha, sp(sigma_alpha))
    beta = pyro.sample("beta", dist.halfcauchy, mu_beta, sp(sigma_beta))

# TODO: training loop
# transformed data
epochs = 100
for i in range(epochs):
    svi = SVI(model, guide, Adam({"lr": 0.001}), "ELBO")
    loss = svi.step(y)
    print("loss = {}").format(loss)

