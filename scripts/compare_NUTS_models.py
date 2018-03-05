from utils import load_data, import_by_string, exists_p, load_p, save_p
import pystan
import numpy as np
import pyro.poutine as poutine
from pdb import set_trace as bb
import pyro
import torch
import pyro.distributions as dist
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc.mcmc import MCMC
#TODO: init_values is a dictionary mapping variable name to values/floats

def run_stan(data, sfile, init_values, n_samples, model_cache=None):
    if model_cache is not None and exists_p(model_cache):
        sm = load_p(model_cache)
    else:
        with open(sfile, "r") as f:
                code = f.read()
        sm = pystan.StanModel(model_code=code)
        if model_cache is not None:
            save_p(sm, model_cache)
    fit = sm.sampling(data=data,iter=n_samples,chains=1,algorithm="NUTS")
    print(fit)

    site_values=fit.extract(permuted=True)
    bb()
    return {k: np.mean(site_values[k]) for k in site_values.keys()}

def run_pyro(data, pfile, n_samples, params):

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


def compare_models(dfile, sfile, pfile, n_samples=1000, model_cache=None):
    data = load_data(dfile)

    params ={}
    import_by_string(pfile + ".init_params")(params)
    print(params)
    s_site_means = run_stan(data,sfile, params, n_samples, model_cache)
    p_site_means = run_pyro(data, pfile, n_samples, params)
    bb()

"""
python -m pdb compare_NUTS_models.py -d ./test/p1/c_p1.json -s ./test/p1/p1.stan -p test.p1.p1 -mc ./test/p1/p1.pkl
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Stan and Pyro models!")
    parser.add_argument('-d', '--data-file', required=True, type=str, help="JSON data file(s)")
    parser.add_argument('-s', '--stan-model-file', required=True, type=str, help="stan model file")
    parser.add_argument('-p', '--pyro-model-file', required=True, type=str, help="Pyro model import path e.g. test.p1")
    parser.add_argument('-mc', '--model-cache', default=None, type=str, help="Stan model cache file")
    parser.add_argument('-ns', '--num-samples', default=1000, type=int, help="number of samples / iterations for HMC")

    args = parser.parse_args()
    compare_models(args.data_file, args.stan_model_file,
                   args.pyro_model_file, model_cache=args.model_cache, n_samples=args.num_samples)