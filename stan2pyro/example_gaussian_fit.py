from pdb import set_trace as bb
import json
with open("normal_100_55.json", "r") as f:
    y_list = json.load(f)

data = {'J': len(y_list),
        'y': y_list}


# import some dependencies
import torch
from torch.autograd import Variable
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam

import collections

def t(x):
    if isinstance(x, collections.Iterable):
        return Variable(torch.FloatTensor(x))
    else:
        return Variable(torch.FloatTensor([x]))


num_epochs = 20


mu = Variable(torch.FloatTensor([0]), requires_grad=True)
sigma = Variable(torch.FloatTensor([1]), requires_grad=True)
        

def model(data):
    #register params
    pyro.param("mu", mu)
    pyro.param("sigma", sigma)

    pyro.sample("y", dist.normal,  mu, sigma, obs=data)

def guide(data=None):
    pass

# setup the optimizer
adam_params = {"lr": 0.05}
optimizer = Adam(adam_params)

loss = SVI(model, guide, optimizer, loss="ELBO")
val = 0
for epoch in range(num_epochs):
    for i in range(data['J']):
        val = loss.step(t(data['y'][i]))
    print("epoch %d loss %0.2f mu %0.2f sigma %0.2f" %
          (epoch+1, val, mu.data[0], sigma.data[0]))

print(mu, sigma)


import pystan

code = """
data {
    int<lower=0> J; // number of data points
    real y[J]; 
}
parameters {
    real mu;
    real<lower=0> sigma;
}
model {
    y ~ normal(mu, sigma);
}
"""


sm = pystan.StanModel(model_code=code)
bb()
fit = sm.sampling(data=data, iter=1000, chains=4)
print(fit)
