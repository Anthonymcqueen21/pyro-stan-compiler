import numpy as np
from compare_models import compare_models
from utils import dist, EPSILON, set_seed, to_variable, to_float, dist, get_fns_pyro, \
    fma, init_real_and_cache, do_pyro_compatibility_hacks, generate_pyro_file, \
    mkdir_p, load_data, json_file_to_mem_format, _pyro_sample, _call_func
import pyro
import torch
import os
import json

from divide_stan_data import divide_json_data
def test1():
    model_cache = "./test/model_1.stan.pkl"
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



def test3():
    n_samples = 1
    model_cache = "./test/model_3.stan.pkl"
    dfile = "../example-models/bugs_examples/vol1/rats/rats.data.R"
    mfile = "../example-models/bugs_examples/vol1/rats/rats_vec.stan"
    pfile = "./test/model_3_autogen.py"
    test_generic(dfile, mfile, pfile, n_samples, model_cache)

def test_generic(dfile, mfile, pfile, n_samples, model_cache):
    generate_pyro_file(mfile, pfile)
    jfile = "./test/model_3.data.json"

    if not os.path.exists(jfile):
        os.system("Rscript --vanilla convert_data.R %s %s" % (dfile, jfile))
    if not os.path.exists(jfile):
        assert False, "R data to json conversion failed"
    with open(jfile, "r") as fj:
        file_data = json.load(fj)
    b, jd1, jd2 = divide_json_data(file_data)
    assert b, "could not divide json data"
    datas = [json_file_to_mem_format(jd1), json_file_to_mem_format(jd2)]

    init_params, model, transformed_data = get_fns_pyro(pfile)

    with open(mfile, "r") as f:
        code = f.read()
    # TODO: remember to coordinate this with Pyro dist = DIST() object in units.py
    code = do_pyro_compatibility_hacks(code)

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

    def transformed_data(data):
        N = data["N"]
        T = data["T"]
        x = data["x"]
        y = data["y"]
        xbar = data["xbar"]
        x_minus_xbar = torch.zeros(T)
        for t in range(1, T + 1):
            x_minus_xbar[t - 1] = to_float((x[t - 1] - xbar))
        data["x_minus_xbar"] = x_minus_xbar
        y_linear = torch.zeros((N * T))
        for n in range(1, N + 1):
            for t in range(1, T + 1):
                y_linear[(((n - 1) * T) + t) - 1] = to_float(y[n - 1][t - 1])
        data["y_linear"] = y_linear

    def init_params(data, params):
        N = data["N"]
        T = data["T"]
        x = data["x"]
        y = data["y"]
        xbar = data["xbar"]
        params["alpha"] = init_real_and_cache("alpha", dims=(N))  # real/double
        params["beta"] = init_real_and_cache("beta", dims=(N))  # real/double
        params["mu_alpha"] = init_real_and_cache("mu_alpha")  # real/double
        params["mu_beta"] = init_real_and_cache("mu_beta")  # real/double
        params["sigmasq_y"] = init_real_and_cache("sigmasq_y", low=0)  # real/double
        params["sigmasq_alpha"] = init_real_and_cache("sigmasq_alpha", low=0)  # real/double
        params["sigmasq_beta"] = init_real_and_cache("sigmasq_beta", low=0)  # real/double

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
        # {
        pred = torch.zeros((N * T))

        for n in range(1, N + 1):
            for t in range(1, T + 1):
                pred[(((n - 1) * T) + t) - 1] = to_float(
                    _call_func("fma", [beta[n - 1], x_minus_xbar[t - 1], alpha[n - 1]]))
        mu_alpha = _pyro_sample(mu_alpha, "mu_alpha", "normal", (to_variable(0), to_variable(100)))
        mu_beta = _pyro_sample(mu_beta, "mu_beta", "normal", (to_variable(0), to_variable(100)))
        sigmasq_y = _pyro_sample(sigmasq_y, "sigmasq_y", "inv_gamma", (to_variable(0.001), to_variable(0.001)))
        sigmasq_alpha = _pyro_sample(sigmasq_alpha, "sigmasq_alpha", "inv_gamma",
                                     (to_variable(0.001), to_variable(0.001)))
        sigmasq_beta = _pyro_sample(sigmasq_beta, "sigmasq_beta", "inv_gamma", (to_variable(0.001), to_variable(0.001)))
        sigma_alpha = _call_func("sqrt", [sigmasq_alpha])
        alpha = _pyro_sample(alpha, "alpha", "normal", (mu_alpha, sigma_alpha))
        sigma_beta = _call_func("sqrt", [sigmasq_beta])
        beta = _pyro_sample(beta, "beta", "normal", (mu_beta, sigma_beta))
        sigma_y = _call_func("sqrt", [sigmasq_y])
        y_linear = _pyro_sample(y_linear, "y_linear", "normal", (pred, sigma_y), obs=y_linear)
        # }

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
    test3()

if __name__ == "__main__":
    test_all()