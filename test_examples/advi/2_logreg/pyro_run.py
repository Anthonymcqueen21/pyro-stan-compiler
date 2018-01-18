# import some dependencies
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam
from utils import to_variable, load_data, do_analysis, fudge
from pdb import set_trace as bb

# hyperparameters - TODO: make them arg parsed with defaults
# assuming SVI
num_epochs = 100
lr = 0.0001

model_name = "2_logreg"

# load data
data = load_data("model.data.json")

def constrain(var,low,high):
    scaled_var = fudge((1.0/(torch.exp(-var) +1)))
    return low + (scaled_var * (high-low))

# model definition
def model(y):
    # sample top level params
    sigma_a = constrain(pyro.sample("sigma_a", dist.normal, to_variable(0), to_variable(1)), 0, 10)     
    sigma_b = constrain(pyro.sample("sigma_b", dist.normal, to_variable(0), to_variable(1)), 0, 10) 
    sigma_c = constrain(pyro.sample("sigma_c", dist.normal, to_variable(0), to_variable(1)), 0, 10) 
    sigma_d = constrain(pyro.sample("sigma_d", dist.normal, to_variable(0), to_variable(1)), 0, 10) 
    sigma_e = constrain(pyro.sample("sigma_e", dist.normal, to_variable(0), to_variable(1)), 0, 10) 
    
    mu_a = Variable(torch.zeros(data['n_age']))
    mu_b = Variable(torch.zeros(data['n_edu']))
    mu_c = Variable(torch.zeros(data['n_age_edu']))
    mu_d = Variable(torch.zeros(data['n_state']))
    mu_e = Variable(torch.zeros(data['n_region_full']))

    assert sigma_a.data[0] > 0
    a = pyro.sample("a", dist.normal, mu_a, sigma_a.expand(data['n_age']))
    b = pyro.sample("b", dist.normal, mu_b, sigma_b.expand(data['n_edu']))
    c = pyro.sample("c", dist.normal, mu_c, sigma_c.expand(data['n_age_edu']))
    d = pyro.sample("d", dist.normal, mu_d, sigma_d.expand(data['n_state']))
    e = pyro.sample("e", dist.normal, mu_e, sigma_e.expand(data['n_region_full']))

    beta = pyro.sample("beta", dist.normal, Variable(torch.zeros(5)), Variable(torch.ones(5)) * 100.)
    
    y_hat =  beta[0].expand(data["N"]) + beta[1].expand(data["N"]) * to_variable(data["black"]) \
             + beta[2].expand(data["N"]) * to_variable(data["female"]) \
             + beta[4].expand(data["N"]) * to_variable(data["female"]) * to_variable(data["black"]) \
             + beta[3] * to_variable(data["v_prev_full"]) + a.index_select(0, to_variable(data["age"]).long() -1) \
             + b.index_select(0, to_variable(data["edu"]).long()-1) \
             + c.index_select(0, to_variable(data["age_edu"]).long()-1) \
             + d.index_select(0, to_variable(data["state"]).long()-1) \
             + e.index_select(0, to_variable(data["region_full"]).long()-1)
    #print(y_hat[0:5])     
    #bb()    
    pyro.sample("y_hat",dist.Bernoulli(ps=fudge(1.0/( torch.exp(-y_hat) + 1))), obs=y)


## guide synthesis
def guide(y):
    # declare toplevel params
    mu_sigma_a = pyro.param("mu_sigma_a", Variable(torch.randn(1), requires_grad=True))
    sigma_sigma_a = pyro.param("sigma_sigma_a", Variable(torch.rand(1), requires_grad=True))
    sigma_sigma_a = constrain(sigma_sigma_a, 0, 10)
    
    mu_sigma_b = pyro.param("mu_sigma_b", Variable(torch.randn(1), requires_grad=True))
    sigma_sigma_b = pyro.param("sigma_sigma_b", Variable(torch.rand(1), requires_grad=True))
    sigma_sigma_b = constrain(sigma_sigma_b, 0, 10)
    
    mu_sigma_c = pyro.param("mu_sigma_c", Variable(torch.randn(1), requires_grad=True))
    sigma_sigma_c = pyro.param("sigma_sigma_c", Variable(torch.rand(1), requires_grad=True))
    sigma_sigma_c = constrain(sigma_sigma_c, 0, 10)
    
    mu_sigma_d = pyro.param("mu_sigma_d", Variable(torch.randn(1), requires_grad=True))
    sigma_sigma_d = pyro.param("sigma_sigma_d", Variable(torch.rand(1), requires_grad=True))
    sigma_sigma_d = constrain(sigma_sigma_d, 0, 10)
    
    
    mu_sigma_e = pyro.param("mu_sigma_e", Variable(torch.randn(1), requires_grad=True))
    sigma_sigma_e = pyro.param("sigma_sigma_e", Variable(torch.rand(1), requires_grad=True))
    sigma_sigma_e = constrain(sigma_sigma_e, 0, 10)
    

    # sample top level params
    sigma_a = constrain(pyro.sample("sigma_a", dist.normal, mu_sigma_a, sigma_sigma_a), 0, 10)
    sigma_b = constrain(pyro.sample("sigma_b", dist.normal, mu_sigma_b, sigma_sigma_b), 0, 10)
    sigma_c = constrain(pyro.sample("sigma_c", dist.normal, mu_sigma_c, sigma_sigma_c), 0, 10)
    sigma_d = constrain(pyro.sample("sigma_d", dist.normal, mu_sigma_d, sigma_sigma_d), 0, 10)
    sigma_e = constrain(pyro.sample("sigma_e", dist.normal, mu_sigma_e, sigma_sigma_e), 0, 10)

    

    # decalre domain level params
    mu_a = pyro.param("mu_a", Variable(torch.randn(data['n_age']), requires_grad=True))
    mu_b = pyro.param("mu_b", Variable(torch.randn(data['n_edu']), requires_grad=True))
    mu_c = pyro.param("mu_c", Variable(torch.randn(data['n_age_edu']), requires_grad=True))
    mu_d = pyro.param("mu_d", Variable(torch.randn(data['n_state']), requires_grad=True))
    mu_e = pyro.param("mu_e", Variable(torch.randn(data['n_region_full']), requires_grad=True))

    # sample domain level params 
    a = pyro.sample("a", dist.normal, mu_a, sigma_a.expand(data['n_age']))
    b = pyro.sample("b", dist.normal, mu_b, sigma_b.expand(data['n_edu']))
    c = pyro.sample("c", dist.normal, mu_c, sigma_c.expand(data['n_age_edu']))
    d = pyro.sample("d", dist.normal, mu_d, sigma_d.expand(data['n_state']))
    e = pyro.sample("e", dist.normal, mu_e, sigma_e.expand(data['n_region_full']))
    
    
    # declare beta params
    mu_beta = pyro.param("mu_beta", Variable(torch.randn(5), requires_grad=True))
    sigma_beta = pyro.param("sigma_beta", Variable(torch.rand(5), requires_grad=True))

    beta = pyro.sample("beta", dist.normal, mu_beta, sigma_beta)
    

torch.manual_seed(0)    
# run inference
# choose optimizer? start with Adam
# setup the optimizer
adam_params = {"lr": lr}
optimizer = Adam(adam_params)

loss = SVI(model, guide, optimizer, loss="ELBO")
val = 0
for epoch in range(num_epochs):
    #TODO: batching if the data is large
    val = loss.step(to_variable(data['y']))
    print('epoch loss: {}'.format(val))

# generate code for printing final params
bb()
# %%%final_params_print%%%

# call problem-specific analysis function
#do_analysis(model_name)


