import numpy as np
from compare_models import compare_models
from utils import  EPSILON, set_seed, to_variable, to_float, get_fns_pyro, \
    fma, init_real, do_pyro_compatibility_hacks, generate_pyro_file, handle_error, \
    mkdir_p, load_data, json_file_to_mem_format, _pyro_sample, _call_func, log_traceback, \
    init_vector, init_matrix, to_int, _pyro_assign, _index_select
import pyro
import pyro.distributions as dist
import torch
import os
import json
import sys
from pdb import set_trace as bb
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
    data ={'N' : N, 'y' : list(0.5 + np.random.randn(N)*0.5)}

    transformed_data = None

    def init_params(data, params):
        N = data["N"]
        y = data["y"]
        params["mu"] =  init_real("mu")
        params["sigma"] = init_real("sigma",low=0.)

    def model(data, params):
        N = data["N"]
        y = data["y"]
        mu = params["mu"]
        sigma = params["sigma"]
        pyro.sample("y", dist.Normal(mu,sigma), obs =y)

    compare_models(code, data, init_params, model, transformed_data, n_runs=2, model_cache=model_cache)

def test5():
    model_cache = "./test/model_5.stan.pkl"

    N=1
    M=1
    data = {
        "y" : 1
    }

    code = """
    data {
      int<lower=0,upper=1> y;
    }
    parameters {
      vector[3] beta;
    }
    transformed parameters {
      real n_avoid;
      real n_shock;
      real p;
      
      n_avoid = 0;
      n_shock = 0;
      p = beta[1] + beta[2] * n_avoid + beta[3] * n_shock;
      
    }
    model {
      beta ~ normal(0, 100);
      y ~ bernoulli_logit(p);
    }
    """
    def init_params(data, params):
        y = data["y"]
        params["beta"] = torch.rand(3)

    def model(data, params):
        y = data["y"]
        beta = params["beta"]
        # INIT transformed parameters
        n_avoid = torch.zeros(1)  # matrix
        n_shock = torch.zeros(1)  # matrix
        p = torch.zeros(1)  # matrix


        n_avoid = torch.zeros(1)
        n_shock = torch.zeros(1)
        p = beta[0] + beta[1] * n_avoid + beta[2] * n_shock

        beta = _pyro_sample(beta, "beta", "normal", [0, 100])


        y = _pyro_sample(y,"y", "bernoulli_logit",p, obs=y)



    m, e = compare_models(code, data, init_params, model, None, n_runs=2, model_cache=model_cache)
    print(m, e)

def test4():
    model_cache = "./test/model_4.stan.pkl"
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
        transformed parameters {
            real mu_p;
            mu_p = -200*mu;
        }
        model {
            mu ~ normal(0,100);
            y ~ normal(mu_p, sigma);
        }
    """
    N=1
    data = {'N': N, 'y': [1]*N}


    transformed_data = None

    def init_params(data, params):
        N = data["N"]
        y = data["y"]
        params["mu"] =  init_real("mu")
        params["sigma"] = init_real("sigma",low=0.)

    def model(data, params):
        N = data["N"]
        y = data["y"]
        mu = params["mu"]
        sigma = params["sigma"]

        mu_p = -200. * mu

        mu = pyro.sample("mu", dist.Normal(torch.zeros(N), torch.ones(N)*100.))

        pyro.sample("y", dist.Normal(mu_p,sigma), obs =y)

    m, e = compare_models(code, data, init_params, model, transformed_data, n_runs=2, model_cache=model_cache)
    print(m, e)
    assert m == 0, "test not successful"

def test3():
    n_samples = 1
    model_cache = "./test/model_3.stan.pkl"
    dfile = "../example-models/bugs_examples/vol1/rats/rats.data.R"
    mfile = "../example-models/bugs_examples/vol1/rats/rats_vec.stan"
    pfile = "./test/model_3_autogen.py"
    m, e = test_generic(dfile, mfile, pfile, n_samples, model_cache)
    print(m,e)
    assert m==0, "test not successful"



from utils import exists_p, load_p, save_p
import pystan
import multiprocessing

def cache_model(args):
    (i,pfile,mfile, model_cache) = args
    idd = "%d - %s" % (i, pfile)
    with open(mfile, "r") as f:
        code = f.read()

    if "increment_log_prob" in code:
        msg = ("T %s: has increment_log_prob" % idd)
        return msg, 2
    assert model_cache is not None
    if exists_p(model_cache):
        msg = ("T %s: exists" % idd)
        return msg, 1
    else:
        try:
            sm = pystan.StanModel(model_code=code)
        except ValueError:
            msg = ("T %s: error in compilation" % idd)
            return msg, 3
        except:
            raise
        save_p(sm, model_cache)
        msg = ("T %s: saved" % idd)
        return msg, 0

def cache_all_models(args):
    i=0
    argss = []
    for (dfile, mfile, pfile, model_cache) in args:
        #print("processing #%d" %i)
        i+=1
        argss.append((i,pfile,mfile, model_cache))

    p = multiprocessing.Pool(27)
    results = p.map(cache_model, argss)
    for k in range(4):
        n = len(list(filter(lambda x: x[1] == k, results)))
        print("%d : %d" % (k, n))

def test_generic(dfile, mfile, pfile, n_runs, model_cache):
    try:
        generate_pyro_file(mfile, pfile)
    except AssertionError as e:
        return handle_error("generate_pyro_file", e)

    jfile = "%s.json" % pfile

    if not os.path.exists(jfile):
        os.system("Rscript --vanilla convert_data.R %s %s" % (dfile, jfile))
    if not os.path.exists(jfile):
        return 1, "R data to json conversion failed"
    with open(jfile, "r") as fj:
        file_data = json.load(fj)
    """
    try:
        b, jd1, jd2 = divide_json_data(file_data)
    except AssertionError as e:
        if "variable values in data.R file are nested dictionaries!" in str(e):
            _, _, etb = sys.exc_info()
            return 14, log_traceback(e, etb)
        raise

    if not b:
        print("Could not divide json data")
        return 2
    datas = [json_file_to_mem_format(jd1), json_file_to_mem_format(jd2)]
    """
    try:
        data = json_file_to_mem_format(file_data)
    except AssertionError as e:
        return handle_error("json_file_to_mem_format", e)

    try:
        validate_data_def, init_params, model, transformed_data = get_fns_pyro(pfile)
    except SyntaxError as e:
        return handle_error("import_pyro_code", e)
    #except AttributeError as e: #one of the attributes/functions was not found in pyro code
    #    return 3

    try:
        validate_data_def(data)
    except (AssertionError, KeyError) as e:
        return handle_error("validate_data_def", e)

    """for data in datas:
        try:
            validate_data_def(data)
        except (AssertionError, KeyError) as e:
            _, _, etb = sys.exc_info()
            return 16, "splitted data validation failed: %s" % log_traceback(e, etb)
        except:
            raise
    """

    try:
        assert model is not None, "model is None"
        assert init_params is not None, "init_params is None"
    except AssertionError as e:
        return handle_error("import_pyro_code", e)

    #if transformed_data is None:
    #    return 5

    with open(mfile, "r") as f:
        code = f.read()

    try:
        assert  "increment_log_prob" not in code, "increment_log_prob used in Stan code"
    except AssertionError as e:
        return handle_error("load_stan_code", e)

    matched, err_s = compare_models(code, data, init_params, model, transformed_data,
                   n_runs=n_runs, model_cache=model_cache)
    return matched, err_s

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
    test5()