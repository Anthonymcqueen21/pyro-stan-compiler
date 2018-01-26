# import some dependencies
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam
from utils import to_variable, load_data, do_analysis, fudge, SmoothedUniform
from pdb import set_trace as bb

# hyperparameters - TODO: make them arg parsed with defaults
# assuming SVI
num_epochs = 100
lr = 0.001
B = 200 # batching size

model_name = "2_logreg"

if __name__ == "__main__":
    # load data
    data = load_data("model.data.json")

def constrain(var,low,high):
    scaled_var = fudge((1.0/(torch.exp(-var) +1)))
    return low + (scaled_var * (high-low))


variables = ["a", "b", "c", "d", "e"]
n_keys = ["n_age", "n_edu", "n_age_edu", "n_state", "n_region_full"]

def compute_y_hat(beta, rvs, B, black, female, v_prev_full, age, edu, age_edu, state, region_full):
    y_hat = beta[0].expand(B) + beta[1].expand(B) * black
    y_hat = y_hat + beta[2].expand(B) * female
    y_hat = y_hat + beta[4].expand(B) * female * black
    y_hat = y_hat + beta[3] * v_prev_full

    y_hat = y_hat + rvs["a"].index_select(0, age.long() - 1)
    y_hat = y_hat + rvs["b"].index_select(0, edu.long() - 1)
    y_hat = y_hat + rvs["c"].index_select(0, age_edu.long() - 1)
    y_hat = y_hat + rvs["d"].index_select(0, state.long() - 1)
    y_hat = y_hat + rvs["e"].index_select(0, region_full.long() - 1)
    return y_hat

MAX_NUM = 100000
# model definition
def model(B,black, female, v_prev_full, age, edu, age_edu, state, region_full, y):

    # sample top level params
    sigmas, mus, rvs = {}, {}, {}
    for i in range(len(variables)):
        v = variables[i]
        sigmas[v] = pyro.sample("_sigma_%s" % v, SmoothedUniform(to_variable(0), to_variable(100)) )
        mus[v] = Variable(torch.zeros(data[n_keys[i]]))
        rvs[v] = pyro.sample(v, dist.normal, mus[v], sigmas[v].expand(data[n_keys[i]]))
    
    beta = pyro.sample("beta", dist.normal, Variable(torch.zeros(5)), Variable(torch.ones(5)) * 100.)
    
    y_hat =  compute_y_hat(beta, rvs, B, black, female, v_prev_full, age, edu, age_edu, state, region_full)

    pyro.sample("y_hat",dist.Bernoulli(ps=fudge(1.0/( torch.exp(-y_hat) + 1))), obs=y)
    return y_hat

if __name__ == "__main__":
    params = {}
    for i in range(len(variables)):
        v = variables[i]
        params["_mu_sigma_%s" % v] = Variable(torch.rand(1)*10., requires_grad=True)
        params["_sigma_sigma_%s" % v] = Variable(torch.rand(1)*5., requires_grad=True)
    params["mu_beta"] = Variable(torch.randn(5)*2., requires_grad=True)
    params["sigma_beta"] = Variable(torch.rand(5)*2., requires_grad=True)

## guide synthesis
def guide(B, black, female, v_prev_full, age, edu, age_edu, state, region_full, y=None):

    # sample top level params
    mu_sigmas, sigma_sigmas, sigmas, mus, rvs = {}, {}, {}, {}, {}
    for i in range(len(variables)):
        v = variables[i]
        mu_sigmas[v] = pyro.param("_mu_sigma_%s" % v, params["_mu_sigma_%s" % v])
        sigma_sigmas[v] = pyro.param("_sigma_sigma_%s" % v,params["_sigma_sigma_%s" % v] )
        #sigma_sigmas[v] = sigma_sigmas[v] * sigma_sigmas[v]
        
        sigmas[v] = pyro.sample("_sigma_%s" % v, dist.lognormal, mu_sigmas[v] ,sigma_sigmas[v])
        #sigmas[v] = torch.abs(sigmas[v])
        mus[v] = Variable(torch.randn(data[n_keys[i]]), requires_grad=True)
        rvs[v] = pyro.sample(v, dist.normal, mus[v], sigmas[v].expand(data[n_keys[i]]))
    
    # declare beta params
    mu_beta = pyro.param("mu_beta", params["mu_beta"])
    sigma_beta = pyro.param("sigma_beta", params["sigma_beta"])
    sigma_beta = torch.abs(sigma_beta)
    
    beta = pyro.sample("beta", dist.normal, mu_beta, sigma_beta)
    y_hat = compute_y_hat(beta, rvs, B, black, female, v_prev_full, age, edu, age_edu, state, region_full)
    return y_hat
def get_data_indices(d, ixs):
    r = []
    for i in ixs:
        r.append(d[i])
    return r
   
data_keys = ["black", "female", "v_prev_full", "age", "edu", "age_edu", "state", "region_full", "y"]
 
if __name__ == "__main__":
    torch.manual_seed(0)    
    # run inference
    # choose optimizer? start with Adam
    # setup the optimizer
    adam_params = {"lr": lr}
    optimizer = Adam(adam_params)

    loss = SVI(model, guide, optimizer, loss="ELBO")
    val = 0
    

    n_train = 10000
    n_test = data["N"] - n_train
    for epoch in range(num_epochs):
        #TODO: batching if the data is large
        indices = range(n_train)
        np.random.shuffle(indices)

        num_batches = 10000/ B
        val = 0
        for j in range(num_batches):
             args = list(map(lambda k: to_variable(get_data_indices(data[k],indices[j*B:j*B+B])), data_keys))
             val += loss.step(B, *args)
        print('epoch %d loss: %0.5f' % (epoch, val))

    args = list(map(lambda k: to_variable(data[k][10000:]), data_keys))
    B = len(args[0])
    p = Variable(torch.zeros(B))
    L = 100000
    for k in range(L):
        y_hat = guide(B, *args)
        p = p + 1.0/( torch.exp(-y_hat) + 1)

    p = ((p / (1. * L)) > 0.5)
    y = args[-1].byte()
    acc =  sum(args[-1].byte() == p).data[0] / (1. * B)

    print("accuracy = %0.5f percent" % (acc*100.))
    bb()
    # %%%final_params_print%%%

    # call problem-specific analysis function
    #do_analysis(model_name)


