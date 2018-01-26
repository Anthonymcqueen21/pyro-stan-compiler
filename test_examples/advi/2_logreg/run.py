from pdb import set_trace as bb 
import pystan
import json
import os
from utils import load_data
from pyro_run import get_data_indices, data_keys
import numpy as np

def separate_train_test(data):
    n_train = 10000
    n_test = data["N"] - n_train
    indices = range(n_train)
    np.random.shuffle(indices)
    train = {}
    for k in data:
        train[k] = data[k]
        if type(data[k]) == list:
            train[k] = get_data_indices(data[k], indices)
    train['N'] = n_train
    test = {}
    for k in data:
        test[k] = data[k]
        if type(data[k]) == list:
            test[k] = data[k][n_train:]
    test['N'] = n_test
    return train, test
    

def run_stan_advi(model_fname, data_fname, iters=1000):
    print("running stan advi script")
    data = load_data(data_fname)
    train, test = separate_train_test(data)
    print("loaded data with %d params = %s" % (len(data), data.keys()))
    with open(model_fname,"r") as f:
        code = f.read()
        
    sm = pystan.StanModel(model_code=code)
    #fit = sm.vb(data=train, iter=iters)
    fit = sm.sampling(data=train, iter=100, algorithm='Fixed_param')
    # print(fit)
    #  fit.keys()
    # ['args', 'inits', 'sampler_params', 'sampler_param_names', 'mean_pars']
    # fit['args'] has arguments for the ADVI algorithm
    # fit['sampler_params'] has as many samples for each param as #iters
    # fit['inits'] ??
    # fit['sampler_param_names'] has all names of parameters
    # fit['mean_pars'] has all learned means
    #la = fit.extract(permuted=True)
    bb()
    
    


if not os.path.exists("model.data.json"):
    print("running Rscript")
    os.system("Rscript transform.R")
    
run_stan_advi("model.stan", "model.data.json")
