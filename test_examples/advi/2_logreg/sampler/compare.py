from utils import to_variable, load_data
from pyro_model import model, variables, n_keys, hardcoded_sigmas
import numpy as np
import pystan
import pickle
import os.path

data = load_data("../model.data.json")

from pdb import set_trace as bb

NS=100000 #num samples


vars = ['a', 'b', 'c', 'd', 'e', 'beta', 'sigma_a', 'sigma_b', 'sigma_c', 'sigma_d', 'sigma_e', 'y']
dims = list(map(lambda i: data[n_keys[i]], range(len(n_keys)))) + [5] + [1]*len(n_keys) + [data["N"]]
print(dims, sum(dims))

SAMPLES_FILE = 'samples_%d.pkl' %NS
MODEL_FILE = "model.pkl"
MEANS_FILE = "means_%d.pkl" %NS

def save_p(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_p(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

def exists_p(fname):
    return os.path.exists(fname)

def run_stan():
    if exists_p(MEANS_FILE):
        means = load_p(MEANS_FILE)
    else:
        if exists_p(MODEL_FILE):
            sm = load_p(MODEL_FILE)
            print("Loaded Stan Model")
        else:
            with open("model_sampler.stan", "r") as f:
                code = f.read()
            sm = pystan.StanModel(model_code=code)
            save_p(sm, MODEL_FILE)

        fit = sm.sampling(data=data,iter=NS,warmup=0,chains=1, algorithm="Fixed_param")
        la=fit.extract(permuted=True)
        arrays = {}
        for v in vars:
            arr_v = np.array(la[v])
            arrays[v] = arr_v
        means = []
        for v in vars:
            arr_v = arrays[v] #np.array(la[v])
            assert arr_v.shape[0] == NS, "arrv.shape[0] != NS"
            v_means = np.mean(arr_v,axis=0)
            if v_means.shape == ():
                v_means = np.array([float(v_means)])
            assert v_means.shape[0] >= 1, "v_means.shape[0] < 1"
            means.append(v_means)
        means = np.concatenate(means)
        save_p(means, MEANS_FILE)
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



print("Running Stan")
s_means = run_stan()

P_MEANS_FILE ="pyro_means_%d.pkl" %NS
if exists_p(P_MEANS_FILE):
    p_means = load_p(P_MEANS_FILE)
else:
    print("Running Pyro")
    p_means = run_pyro()
    save_p(p_means, P_MEANS_FILE)


assert p_means.shape == s_means.shape, "p/m means shape mismatch"

EPS=0.01
def check_eq(i,a,b, e=EPS):
    assert abs(a-b) < e, "issue with index=%d p=%0.5f s=%0.5f" % (i,a,b)

assert sum(dims) == len(p_means), "sum sims != len(p_means)"


curr_var_ix = 0
curr_sum_dims = dims[0]
v = vars[curr_var_ix]
i = 0
while i < len(p_means):
    if i >= curr_sum_dims:
        curr_var_ix +=1
        curr_sum_dims += dims[curr_var_ix]
    v = vars[curr_var_ix]
    print ("checking %s.%d" % (v, i - curr_sum_dims + dims[curr_var_ix]))

    #for i in range(len(p_means)):
    check_eq(i, p_means[i], s_means[i],
             e=(hardcoded_sigmas[v]/10. if (v in variables) else (2.5 if v == "beta" else EPS)))
    i+=1
