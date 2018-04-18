
from utils import  EPSILON, set_seed, to_variable, to_float, get_fns_pyro, tensorize_data, \
    fma, init_real, do_pyro_compatibility_hacks, generate_pyro_file, handle_error, \
    mkdir_p, load_data, json_file_to_mem_format, _pyro_sample, _call_func, log_traceback, \
    init_vector, init_matrix, to_int, _pyro_assign, _index_select, import_by_string, exists_p, load_p, save_p
import pystan
import numpy as np
import pyro.poutine as poutine
from pdb import set_trace as bb
import pyro
import torch
import json
import os.path
import pyro.distributions as dist
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc.mcmc import MCMC
from run_compiler_all_examples import get_all_data_paths, get_cached_state
#TODO: init_values is a dictionary mapping variable name to values/floats

def run_stan_nuts(data, sfile, n_samples=2000, model_cache=None):
    if model_cache is not None and exists_p(model_cache):
        sm = load_p(model_cache)
    else:
        with open(sfile, "r") as f:
                code = f.read()
        sm = pystan.StanModel(model_code=code)
        if model_cache is not None:
            save_p(sm, model_cache)
    fit = sm.sampling(data=data,iter=n_samples,algorithm="NUTS")
    print(fit)

    site_values=fit.extract(permuted=True)
    site_keys = site_values.keys()
    return {k: np.mean(site_values[k], axis=0) for k in site_keys}

def run_pyro_nuts(data, pfile, n_samples, params):

    # import model, transformed_data functions (if exists) from pyro module

    model = import_by_string(pfile + ".model")
    assert model is not None, "model couldn't be imported"
    transformed_data = import_by_string(pfile + ".transformed_data")
    if transformed_data is not None:
        transformed_data(data)

    nuts_kernel = NUTS(model, step_size=0.0855)
    mcmc_run = MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=int(n_samples/2))
    posteriors = {k: [] for k in params}

    for trace, _ in mcmc_run._traces(data, params):
        for k in posteriors:
            posteriors[k].append(trace.nodes[k]['value'])


    #posteriors["sigma"] = list(map(torch.exp, posteriors["log_sigma"]))
    #del posteriors["log_sigma"]

    posterior_means = {k: torch.mean(torch.stack(posteriors[k]), 0) for k in posteriors}
    bb()
    return posterior_means


def run_pyro_advi(data, validate_data_def, initialized_params, model):
    raise NotImplementedError("run_pyro_advi: Implement this JP")

def compare_models(data, sfile, pfile, n_samples=2000, model_cache=None):
    validate_data_def, init_params, model, transformed_data = get_fns_pyro(pfile)

    s_site_means = run_stan_nuts(data, sfile, n_samples=n_samples, model_cache=model_cache)
    # breakpoint
    initialized_params = {}
    tensorize_data(data)
    transformed_data(data)
    init_params(data, initialized_params)
    p_site_means = run_pyro_advi(data, validate_data_def, initialized_params, model)

    #TODO: check that these means are close to each other for each parameter

    bb()


def use_log_prob_tested_models(ofldr):
    cfname = "%s/status.pkl" % ofldr
    (j, status) = get_cached_state(cfname)
    successful_models = status[0]
    # args = list(sorted(get_all_data_paths(p_args.examples_folder,ofldr)))
    print("%d =total possible (R data, stan model) pairs that pass log-prob test" % len(successful_models))

    for (dfile, mfile, pfile, model_cache, err) in (successful_models):
        if os.path.exists(pfile):
            jfile = "%s.json" % pfile
            assert os.path.exists(jfile)
            with open(jfile, "r") as fj:
                file_data = json.load(fj)
            data = json_file_to_mem_format(file_data)
            print("STARTING TO PROCESS stan-file: %s pyro-file: %s" % (mfile, pfile))
            compare_models(data, mfile, pfile, model_cache=model_cache, n_samples=2000)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-folder', default='./test_compiler', type=str,
                        help="Output folder from run_compiler_all_examples.py, should have a status.pkl file in it")
    p_args = parser.parse_args()
    use_log_prob_tested_models(p_args.output_folder)


