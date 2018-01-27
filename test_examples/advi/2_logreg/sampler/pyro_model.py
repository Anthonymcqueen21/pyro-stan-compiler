
import torch
from torch.autograd import Variable
import pyro
import pyro.distributions as dist
from utils import to_variable, fudge, SmoothedUniform
from pdb import set_trace as bb


inits_sigmas = {
            "sigma_a" : 20.,
            "sigma_b": 40.,
            "sigma_c": 60.,
            "sigma_d": 80.,
            "sigma_e": 99.
        }


variables = ["a", "b", "c", "d", "e"]
n_keys = ["n_age", "n_edu", "n_age_edu", "n_state", "n_region_full"]
hardcoded_sigmas = {v: inits_sigmas["sigma_%s"%v] for v in variables}

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


# model definition
def model(B, la, black, female, v_prev_full, age, edu, age_edu, state, region_full, y, data):
    # sample top level params
    sigmas, mus, rvs = {}, {}, {}
    for i in range(len(variables)):
        v = variables[i]
        #sigmas[v] = pyro.sample("_sigma_%s" % v, dist.uniform, to_variable(0), to_variable(100))
        sigmas[v] = to_variable(hardcoded_sigmas[v])
        mus[v] = Variable(torch.zeros(data[n_keys[i]]))
        rvs[v] = pyro.sample(v, dist.normal, mus[v], sigmas[v].expand(data[n_keys[i]]))

    beta = pyro.sample("beta", dist.normal, Variable(torch.zeros(5)), Variable(torch.ones(5)) * 100.)

    y_hat = compute_y_hat(beta, rvs, B, black, female, v_prev_full, age, edu, age_edu, state, region_full)

    y = pyro.sample("y", dist.Bernoulli(ps=fudge(1.0 / (torch.exp(-y_hat) + 1))))
    return sigmas, rvs, beta, y


