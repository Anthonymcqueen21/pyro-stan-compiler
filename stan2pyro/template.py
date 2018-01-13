# import some dependencies
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam
from utils import to_variable, load_data


# hyperparameters - TODO: make them arg parsed with defaults
# assuming SVI
num_epochs = 1000 
lr = 0.001

model_name = %%%

# load data
data = load_data(model_name)

# model definition
def model(%%%):
    %%%

## guide synthesis
def guide(%%%):
    # parameter initialization and registering
    %%%
    # generate mean-field guide with pyro.sample statements
    # assume: normal distribution for continuous, categorical for discrete
    %%%
    
# run inference
# choose optimizer? start with Adam
# setup the optimizer
adam_params = {"lr": lr}
optimizer = Adam(adam_params)

loss = SVI(model, guide, optimizer, loss="ELBO")
val = 0
for epoch in range(num_epochs):
    #TODO: batching if the data is large
    val = loss.step(t(data['n']), t(data['y']), t(data['Z']))
    print('epoch loss: {}'.format(val))
