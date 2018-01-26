from utils import to_variable, load_data
from pyro_model import model, variables
import numpy as np
import pystan
import pickle
import os.path

data = load_data("../model.data.json")

from pdb import set_trace as bb

NS=10000 #num samples


vars = ['a', 'b', 'c', 'd', 'e', 'beta', 'sigma_a', 'sigma_b', 'sigma_c', 'sigma_d', 'sigma_e', 'y']
SAMPLES_FILE = 'samples_%d.pkl' %NS

def save_samples(la):
    with open("samples.pkl","wb") as f:
        pickle.dump(la, f)

def load_samples():
    with open(SAMPLES_FILE, 'rb') as f:
        samples = pickle.load(f)
    return samples

def exists_samples():
    return os.path.exists(SAMPLES_FILE)

MODEL_FILE = "model.pkl"
def save_model(sm):
    # save it to the file 'model.pkl' for later use
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(sm, f)

def load_model():
    # save it to the file 'model.pkl' for later use
    with open(MODEL_FILE, 'rb') as f:
        sm = pickle.load(f)
    return sm

def exists_model():
    return os.path.exists(MODEL_FILE)

def run_stan():
    with open("model_sampler.stan","r") as f:
        code = f.read()
    if exists_samples():
        arrays = load_samples()
    else:
        if exists_model():
            sm = load_model()
        else:
            sm = pystan.StanModel(model_code=code)
            save_model(sm)

        fit = sm.sampling(data=data,iter=NS,warmup=0,chains=1, algorithm="Fixed_param")
        la=fit.extract(permuted=True)
        arrays = {}
        for v in vars:
            arr_v = np.array(la[v])
            arrays[v] = arr_v
        save_samples(arrays)

    means = []
    for v in vars:
        arr_v = arrays[v] #np.array(la[v])
        assert arr_v.shape[0] == NS
        v_means = np.mean(arr_v,axis=0)
        if v_means.shape == ():
            v_means = np.array([float(v_means)])
        assert v_means.shape[0] >= 1
        means.append(v_means)
    means = np.concatenate(means)
    return means


def run_pyro():
    data_keys = ["black", "female", "v_prev_full", "age", "edu", "age_edu", "state", "region_full", "y"]

    B=data['N']
    args = list(map(lambda k: to_variable(data[k]), data_keys))
    args.append(data)

    vals = []
    for j in range(NS):
        sigmas, rvs, beta, y_hat = model(B,*args)
        val = np.concatenate([rvs[v].data.numpy() for v in variables] + [beta.data.numpy()] +
                       [sigmas[v].data.numpy() for v in variables] + [y_hat.data.numpy()])
        vals.append(val)

    arr = np.array(vals)
    means = np.mean(arr, axis=0)
    return means

p_means = run_pyro()
s_means = run_stan()

assert p_means.shape == s_means.shape

EPS=0.1
def check_eq(i,a,b):
    if a == 0.:
        assert abs(b) < EPS, "issue with index=%d p=%0.5f s=%0.5f" % (i,a,b)
    else:
        assert abs(a-b)/abs(a) < EPS, "issue with index=%d p=%0.5f s=%0.5f" % (i,a,b)

for i in range(len(p_means)):
    check_eq(i, p_means[i], s_means[i])

