# import some dependencies
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam
from utils import to_variable, load_data, do_analysis


# hyperparameters - TODO: make them arg parsed with defaults
# assuming SVI
num_epochs = 1000 
lr = 0.001

model_name = %%%model_name%%%

# load data
data = load_data(model_name)

# model definition
def model(%%%model_args%%%):
    %%%model_def%%%

## guide synthesis
def guide(%%%model_args%%%):
    # parameter initialization and registering
    %%%init_params%%%
    # generate mean-field guide with pyro.sample statements
    # assume: normal distribution for continuous, categorical for discrete
    %%%mean_field_guide%%%
    
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

# generate code for printing final params
%%%final_params_print%%%

# call problem-specific analysis function
do_analysis(model_name)
