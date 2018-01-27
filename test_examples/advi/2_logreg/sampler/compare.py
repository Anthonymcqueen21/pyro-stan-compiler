from pyro_model import model, variables, n_keys, hardcoded_sigmas, inits_sigmas, compute_y_hat
import pyro
import pyro.poutine as poutine
import torch
import numpy as np
import pystan
import pickle
import os.path
import pyro.distributions as dist
from utils import to_variable, load_data, fudge
from pdb import set_trace as bb

data = load_data("../model.data.json")
NS=10 #num samples
vars = ['a', 'b', 'c', 'd', 'e', 'beta', 'sigma_a', 'sigma_b', 'sigma_c', 'sigma_d', 'sigma_e', 'y']
dims = list(map(lambda i: data[n_keys[i]], range(len(n_keys)))) + [5] + [1]*len(n_keys) + [data["N"]]
print(dims, sum(dims))
SAMPLES_FILE = 'samples_%d.pkl' %NS
MODEL_FILE = "model.pkl"
LOG_PROBS_FILE = "log_probs_stan_%d.pkl" % NS

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
    if exists_p(LOG_PROBS_FILE):
        (log_probs,la) = load_p(LOG_PROBS_FILE)
    else:
        if exists_p(MODEL_FILE):
            sm = load_p(MODEL_FILE)
            print("Loaded Stan Model")
        else:
            #with open("model_sampler.stan", "r") as f:
            with open("../model.stan", "r") as f:
                    code = f.read()
            sm = pystan.StanModel(model_code=code)
            save_p(sm, MODEL_FILE)

        fit = sm.sampling(data=data,iter=NS,warmup=0,chains=1, init=[inits_sigmas], algorithm="Fixed_param")
        la=fit.extract(permuted=True)
        log_probs = []
        for i in range(NS):
            params = {v: la[v][i] for v in la}
            log_probs.append(fit.log_prob(fit.unconstrain_pars(params)))
        save_p((log_probs,la), LOG_PROBS_FILE)
    return la, log_probs

def guide(B, la, black, female, v_prev_full, age, edu, age_edu, state, region_full, y, data):
    sampled_vals = {}
    for v in vars:
        if v != "y":
            sampled_vals[v] = pyro.sample(v,dist.delta, to_variable(la[v]))
    y_hat = compute_y_hat(sampled_vals["beta"], {v:sampled_vals[v] for v in variables}, B,
                  black, female, v_prev_full, age, edu, age_edu, state, region_full)
    y = pyro.sample("y", dist.Bernoulli(ps=fudge(1.0 / (torch.exp(-y_hat) + 1))))

def run_pyro(la):
    data_keys = ["black", "female", "v_prev_full", "age", "edu", "age_edu", "state", "region_full", "y"]
    B=data['N']
    args = list(map(lambda k: to_variable(data[k]), data_keys))
    args.append(data)
    log_pdfs = []
    for j in range(NS):
        guide_trace = poutine.trace(guide).get_trace(B,{v: la[v][j] for v in la},*args)
        model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                    graph_type="flat").get_trace(B, None, *args)
        log_pdfs.append(model_trace.log_pdf())
    return log_pdfs



print("Running Stan")
la, s_log_probs = run_stan()

print("Running Pyro")
p_log_probs = run_pyro(la)
p_log_probs = [logp.data[0] for logp in p_log_probs]

assert len(p_log_probs) == len(s_log_probs)

p_avg = np.mean(p_log_probs)
s_avg = np.mean(s_log_probs)

if abs(p_avg - s_avg) >= 0.1:
    print("p/m log_probs sum mismatch s=%0.5f p=%0.5f" % (s_avg, p_avg))
else:
    print("It matches!")
bb()