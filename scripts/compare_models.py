from utils import load_data, import_by_string, exists_p, load_p, to_float, save_p, \
    do_pyro_compatibility_hacks, mk_module, tensorize_data, variablize_params, reset_initialization_cache
import pystan
import numpy as np
import pyro.poutine as poutine
from pdb import set_trace as bb
import copy
#TODO: init_values is a dictionary mapping variable name to values/floats

def run_stan(data, code, init_values, n_samples, model_cache=None):
    if model_cache is not None and exists_p(model_cache):
        sm = load_p(model_cache)
    else:
        sm = pystan.StanModel(model_code=code)
        if model_cache is not None:
            save_p(sm, model_cache)
    fit = sm.sampling(data=data,iter=n_samples,warmup=0,chains=1,
                      init=[init_values]
                      if init_values is not None else None, algorithm="Fixed_param")
    site_values=fit.extract(permuted=True)

    log_probs = []
    for i in range(n_samples):
        if n_samples > 1:
            params = {v: site_values[v][i] for v in site_values}
        else:
            params = {v: float(site_values[v]) if  site_values[v].shape == () else site_values[v][0] for v in site_values}
        log_p = fit.log_prob(fit.unconstrain_pars(params), adjust_transform=True)
        #lp2 = fit.log_prob(fit.unconstrain_pars(params), adjust_transform=True)
        log_probs.append(log_p)
        #print(log_p, params, fit.unconstrain_pars(params))
        ### https://github.com/stan-dev/rstan/issues/383
        ### https://groups.google.com/forum/#!topic/stan-users/nmh5oy-Ox9U it is not normalized!
        ### Similar test case as https://github.com/stan-dev/pystan/blob/develop/pystan/tests/test_rstan_stanfit.py#L61
    return site_values, log_probs


def get_num_log_probs(trace):
    n = 0
    for name in trace.nodes:
        if name not in ['_INPUT', '_OUTPUT', '_RETURN']:
            n += np.product(trace.nodes[name]['value'].shape)
    assert n > 0
    return n

def run_pyro(site_values, data, model, transformed_data, n_samples, params):

    # import model, transformed_data functions (if exists) from pyro module

    assert model is not None, "model couldn't be imported"



    variablize_params(params)

    log_pdfs = []
    n_log_probs = None
    for j in range(n_samples):
        if n_samples > 1:
            sample_site_values = {v: site_values[v][j] for v in site_values}
        else:
            sample_site_values = {v: float(site_values[v]) if site_values[v].shape == () else site_values[v][0] for v in
                      site_values}
        #print(sample_site_values)
        variablize_params(sample_site_values)

        model_trace = poutine.trace(poutine.condition(model, data=sample_site_values),
                                    graph_type="flat").get_trace(data, params)
        log_p = model_trace.log_pdf()
        if n_log_probs is None:
            n_log_probs = get_num_log_probs(model_trace)
        else:
            assert n_log_probs == get_num_log_probs(model_trace)
        #print(log_p.data.numpy())
        log_pdfs.append(to_float(log_p))
    return log_pdfs, n_log_probs



def process_files(pfile, dfiles, sfile):
    with open(sfile, "r") as f:
        code = f.read()
    code = do_pyro_compatibility_hacks(code)

    def sanitize_module_loading_file(pfile):
        if pfile.endswith(".py"):
            pfile = pfile[:-3]
        pfile = pfile.replace("/", ".")
        pfile = pfile.strip(".")
        return pfile

    pfile = sanitize_module_loading_file(pfile)
    mk_module(pfile)

    def get_fns_pyro(pfile):
        model = import_by_string(pfile + ".model")
        assert model is not None, "model couldn't be imported"
        transformed_data = import_by_string(pfile + ".transformed_data")
        init_params = import_by_string(pfile + ".init_params")
        return init_params, model, transformed_data

    init_params, model, transformed_data = get_fns_pyro(pfile)

    datas = [load_data(dfile) for dfile in dfiles]

    return code, datas, init_params, model, transformed_data


def compare_models(code, datas, init_params, model, transformed_data, n_samples=1, model_cache=None):
    reset_initialization_cache()
    lp_vals = []
    for data in datas:
        params ={}
        copy_data = copy.deepcopy(data)

        tensorize_data(data)
        if transformed_data is not None:
            try:
                transformed_data(data)
            except KeyError:
                return 8
            except:
                raise
        try:
            init_params(data, params)
        except KeyError:
            return 9

        init_values = {k: params[k].data.cpu().numpy().tolist() for k in params}
        init_values = {k: (init_values[k][0]) if len(init_values[k])==1 else np.array(init_values[k]) for k in init_values}

        #print(init_values)
        site_values, s_log_probs = run_stan(copy_data, code, init_values, n_samples, model_cache)
        p_log_probs, n_log_probs = run_pyro(site_values, data, model, transformed_data, n_samples, params)

        p_avg = (np.mean(p_log_probs))/n_log_probs
        s_avg = (np.mean(s_log_probs))/n_log_probs
        lp_vals.append((p_avg, s_avg))

    assert len(lp_vals) >= 2
    diffs = list(map(lambda x: x[0]-x[1], lp_vals))
    diff0 = diffs[0]
    for diff_v in diffs:

        if abs(diff_v-diff0) >= 1e-2:
            print("Log probs don't match with EPS=1e-2! %s // %s" % (lp_vals, diffs))
            return False
        #print(abs(diff_v-diff0))
    print("Log probs match with a constant difference pyro-stan of approx. %0.3f" % diff0)
    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Stan and Pyro models!")
    parser.add_argument('-ds', '--data-files', required=True, nargs='+', help="JSON data file(s)")
    parser.add_argument('-s', '--stan-model-file', required=True, type=str, help="stan model file")
    parser.add_argument('-p', '--pyro-model-file', required=True, type=str, help="Pyro model import path e.g. test.p1")
    parser.add_argument('-mc', '--model-cache', default=None, type=str, help="Stan model cache file")
    parser.add_argument('-n', '--n-samples', default=1, type=int, help="num samples")

    args = parser.parse_args()
    code, datas, init_params, model, transformed_data = \
        process_files(args.pyro_model_file,args.data_files, args.stan_model_file)
    compare_models(code, datas, init_params, model, transformed_data ,
                   n_samples=args.n_samples, model_cache=args.model_cache)