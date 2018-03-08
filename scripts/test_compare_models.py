import numpy as np
from compare_models import compare_models
from utils import dist, EPSILON, set_seed, to_variable, to_float, dist, \
    fma, init_real_and_cache, do_pyro_compatibility_hacks, \
    mkdir_p, load_data, json_file_to_mem_format
import pyro
import torch
import os
import json

from divide_stan_data import divide_json_data
def test1():
    model_cache = "./test/model_1.pkl"
    n_samples =1
    code = """
        data {
            int N;
            real y[N];
        }
        parameters {
            real mu;
            real<lower=0> sigma;
        }
        model {
            y ~ normal(mu, sigma);
        }
    """
    N=1
    datas = [
        {'N' : N, 'y' : list(0.5 + np.random.randn(N)*0.5)},
        {'N': N, 'y': list(0.5 + np.random.randn(N) * 0.5)},
    ]

    transformed_data = None

    def init_params(data, params):
        N = data["N"]
        y = data["y"]
        params["mu"] =  init_real_and_cache("mu")
        params["sigma"] = init_real_and_cache("sigma",low=0.)

    def model(data, params):
        N = data["N"]
        y = data["y"]
        mu = params["mu"]
        sigma = params["sigma"]
        pyro.sample("y", dist.Normal(mu,sigma), obs =y)

    compare_models(code, datas, init_params, model, transformed_data,
                    n_samples=n_samples, model_cache=model_cache)

def test2():
    n_samples=1
    model_cache = "./test/model_2.stan.pkl"
    dfile = "../example-models/bugs_examples/vol1/rats/rats.data.R"
    jfile = "./test/model_2.data.json"
    if not os.path.exists(jfile):
        os.system("Rscript --vanilla convert_data.R %s %s" % (dfile, jfile))
    if not os.path.exists(jfile):
        assert False, "R data to json conversion failed"
    with open(jfile, "r") as fj:
        file_data = json.load(fj)
    b, jd1, jd2 = divide_json_data(file_data)
    assert b, "could not divide json data"
    datas = [json_file_to_mem_format(jd1), json_file_to_mem_format(jd2)]
    """
    1. remove ";" from end of lines
    2. generate definition for x_minus_xbar, y_linear in transformed_data (done) + pred in model (TODO)
    3. add to_float inside loops / use _assign in python
    4. Add data as argument and initialize data variables in init_params,
    NOTE: don't add transformed_data variables! only those from data block
    5. Replace all inv_gamma( with normal( in stan + pyro -- DONE in python
    6. add () to distributions in init_params
    7. when sampling for expr = pyro.sample -- use expand_as(expr) inside the distribution on params of the distribution
    8. indices used inside loop replace i --> i+1 AND for every indexing operation, add -1!
    9. Deal with addmm translation from fma -- for now using actual math as a lib function fma
    10. Add init_real_and_cache instead of different inits for each time init_params is called - 
    generate dims and low, high kwargs as needed
    """

    def transformed_data(data):
        N = data["N"]
        T = data["T"]
        x = data["x"]
        y = data["y"]
        xbar = data["xbar"]
        x_minus_xbar = torch.zeros(T)  # TODO
        for t in range(0, T):
            x_minus_xbar[t + 1 - 1] = to_float(x[t + 1 - 1] - xbar)  # TODO
        data["x_minus_xbar"] = x_minus_xbar

        y_linear = torch.zeros(N * T)  # TODO
        for n in range(0, N):
            for t in range(0, T):
                y_linear[((n * T) + t + 1 - 1)] = to_float(y[n + 1 - 1][t + 1 - 1])  # TODO
        data["y_linear"] = y_linear

    def init_params(data, params):
        # TODO
        N = data["N"]
        T = data["T"]
        x = data["x"]
        y = data["y"]
        xbar = data["xbar"]
        # TODO: all statements below add () after distributions!
        # TODO: add init_real_and_cache instead of dist.Uniform
        params["alpha"] = init_real_and_cache("alpha", dims=(N))
        params["beta"] = init_real_and_cache("beta", dims=(N))
        params["mu_alpha"] = init_real_and_cache("mu_alpha")
        params["mu_beta"] = init_real_and_cache("mu_beta")
        params["sigmasq_y"] = init_real_and_cache("sigmasq_y", low=0.)
        params["sigmasq_alpha"] = init_real_and_cache("sigmasq_alpha", low=0.)
        params["sigmasq_beta"] = init_real_and_cache("sigmasq_beta", low=0.)

    def model(data, params):
        N = data["N"]
        T = data["T"]
        x = data["x"]
        y = data["y"]
        xbar = data["xbar"]
        x_minus_xbar = data["x_minus_xbar"]
        y_linear = data["y_linear"]
        alpha = params["alpha"]
        beta = params["beta"]
        mu_alpha = params["mu_alpha"]
        mu_beta = params["mu_beta"]
        sigmasq_y = params["sigmasq_y"]
        sigmasq_alpha = params["sigmasq_alpha"]
        sigmasq_beta = params["sigmasq_beta"]
        pred = torch.zeros(N * T)  # TODO
        for n in range(0, N):
            for t in range(0, T):
                pred[(((n) * T) + t + 1 - 1)] = to_float(
                    fma(beta[n + 1 - 1], x_minus_xbar[t + 1 - 1], alpha[n + 1 - 1]))  # TODO
        mu_alpha = pyro.sample("mu_alpha", dist.Normal(to_variable(0), to_variable(100)))
        mu_beta = pyro.sample("mu_beta", dist.Normal(to_variable(0), to_variable(100)))
        sigmasq_y = pyro.sample("sigmasq_y", dist.Inv_gamma(to_variable(0.001), to_variable(0.001)))
        sigmasq_alpha = pyro.sample("sigmasq_alpha", dist.Inv_gamma(to_variable(0.001), to_variable(0.001)))
        sigmasq_beta = pyro.sample("sigmasq_beta", dist.Inv_gamma(to_variable(0.001), to_variable(0.001)))
        sigma_alpha = torch.sqrt(sigmasq_alpha)
        alpha = pyro.sample("alpha", dist.Normal(mu_alpha.expand_as(alpha), sigma_alpha.expand_as(alpha)))  # TODO
        sigma_beta = torch.sqrt(sigmasq_beta)
        beta = pyro.sample("beta", dist.Normal(mu_beta.expand_as(beta), sigma_beta.expand_as(beta)))  # TODO
        sigma_y = torch.sqrt(sigmasq_y)
        y_linear = pyro.sample("y_linear", dist.Normal(pred, sigma_y), obs=y_linear)

    with open("../example-models/bugs_examples/vol1/rats/rats_vec.stan", "r") as f:
        code = f.read()

    code = do_pyro_compatibility_hacks(code)

    compare_models(code, datas, init_params, model, transformed_data,
                   n_samples=n_samples, model_cache=model_cache)
def test_all():
    set_seed(0, False)
    mkdir_p("./test")
    test1()
    test2()

if __name__ == "__main__":
    test_all()