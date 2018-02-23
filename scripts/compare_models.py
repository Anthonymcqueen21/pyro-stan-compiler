from utils import load_data, import_by_string, exists_p, load_p, save_p
import pystan
import numpy as np
import pyro.poutine as poutine
from pdb import set_trace as bb
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
    fit = sm.sampling(data=data,iter=n_samples,warmup=0,chains=1,
                      init=[init_values] if init_values is not None else None, algorithm="Fixed_param")
    site_values=fit.extract(permuted=True)

    log_probs = []
    for i in range(n_samples):
        if n_samples > 1:
            params = {v: site_values[v][i] for v in site_values}
        else:
            params = {v: site_values[v] for v in site_values}
        log_p = fit.log_prob(fit.unconstrain_pars(params), adjust_transform=True)
        #lp2 = fit.log_prob(fit.unconstrain_pars(params), adjust_transform=True)
        log_probs.append(log_p)
        #print(log_p, params, fit.unconstrain_pars(params))
        ### https://github.com/stan-dev/rstan/issues/383
        ### https://groups.google.com/forum/#!topic/stan-users/nmh5oy-Ox9U it is not normalized!
        ### Similar test case as https://github.com/stan-dev/pystan/blob/develop/pystan/tests/test_rstan_stanfit.py#L61
    return site_values, log_probs

def run_pyro(site_values, data, pfile, n_samples, params):

    # import model, transformed_data functions (if exists) from pyro module

    model = import_by_string(pfile + ".model")
    assert model is not None, "model couldn't be imported"
    transformed_data = import_by_string(pfile + ".transformed_data")
    if transformed_data is not None:
        transformed_data(data)
    log_pdfs = []
    for j in range(n_samples):
        if n_samples > 1:
            sample_site_values = {v: site_values[v][j] for v in site_values}
        else:
            sample_site_values = {v: site_values[v] for v in site_values}
        model_trace = poutine.trace(poutine.condition(model, data=sample_site_values),
                                    graph_type="flat").get_trace(data, params)
        log_p = model_trace.log_pdf()

        #print(log_p.data.numpy())
        log_pdfs.append(log_p)
    return log_pdfs


def compare_models(dfiles, sfile, pfile, n_samples=1, model_cache=None):
    lp_vals = []
    for dfile in dfiles:
        data = load_data(dfile)

        params ={}
        import_by_string(pfile + ".init_params")(params)
        print(params)
        site_values, s_log_probs = run_stan(data,sfile, params, n_samples, model_cache)
        p_log_probs = run_pyro(site_values, data, pfile, n_samples, params)

        p_avg = np.mean(p_log_probs)
        s_avg = np.mean(s_log_probs)
        lp_vals.append((p_avg, s_avg))

        """
        if abs(p_avg - s_avg) >= 0.1:
            print("p/m log_probs sum mismatch s=%0.5f p=%0.5f" % (s_avg, p_avg))
        else:
            print("It matches!")
        """
    assert len(lp_vals) >= 2
    diffs = list(map(lambda x: x[0]-x[1], lp_vals))
    diff0 = diffs[0]
    for diff_v in diffs:
        assert abs(diff_v-diff0) < 1e-3, "%s // %s" % (lp_vals, diffs)
    print("Log probs match with a constant difference pyro-stan of approx. %0.3f" % diff0)
    bb()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Stan and Pyro models!")
    parser.add_argument('-ds', '--data-files', required=True, nargs='+', help="JSON data file(s)")
    parser.add_argument('-s', '--stan-model-file', required=True, type=str, help="stan model file")
    parser.add_argument('-p', '--pyro-model-file', required=True, type=str, help="Pyro model import path e.g. test.p1")
    parser.add_argument('-mc', '--model-cache', default=None, type=str, help="Stan model cache file")

    args = parser.parse_args()
    compare_models(args.data_files, args.stan_model_file,
                   args.pyro_model_file, model_cache=args.model_cache)