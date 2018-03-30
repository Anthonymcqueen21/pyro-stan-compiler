import os
import collections
import numpy as np
from os.path import join
import subprocess
from six import string_types
from .logger import log_traceback


EPSILON = 1e-7


def sanitize_module_loading_file(pfile):
    if pfile.endswith(".py"):
        pfile = pfile[:-3]
    pfile = pfile.replace("/", ".")
    pfile = pfile.strip(".")
    return pfile


def generate_pyro_file(mfile, pfile):
    process = subprocess.Popen('../stan2pyro/bin/stan2pyro %s' % mfile, shell=True,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr =subprocess.PIPE, close_fds=True)

    out, err = process.communicate()
    out = out.decode('utf-8')
    if err is None:
        err = ""
    else:
        err = err.decode('utf-8')

    with open(pfile, "w") as f:
        f.write("# model file: %s\n" % mfile)
        f.write("from utils import to_float, _pyro_sample, _call_func, check_constraints\n")
        f.write("from utils import init_real, init_vector, init_matrix, init_int\n")
        f.write("from utils import _index_select, to_int, _pyro_assign, as_bool\n")
        f.write("import torch\nimport pyro\n")
        # TODO remove to_variable
        f.write("from utils import identity as to_variable\n\n")
        f.write(out + "\n")
        for err_s in ["SYNTAX ERROR, MESSAGE(S) FROM PARSER", "Aborted (core dumped)", "SAMPLING CONSTANTS NOT SUPPORTED"]:
            assert err_s not in out and err_s not in err, "SYNTAX ERROR in Stan Code: %s" % err


def get_fns_pyro(pfile):
    pfile = sanitize_module_loading_file(pfile)
    mk_module(pfile)
    model = import_by_string(pfile + ".model") #reqiured
    #assert model is not None, "model couldn't be imported"
    try:
        transformed_data = import_by_string(pfile + ".transformed_data")
    except AttributeError as e:
        transformed_data = None
    init_params = import_by_string(pfile + ".init_params") #reqiured
    validate_data_def = import_by_string(pfile + ".validate_data_def") #reqiured
    return validate_data_def, init_params, model, transformed_data


def tensorize_data(data):
    to_delete = []
    for k in data:
        if isinstance(data[k], float) or isinstance(data[k], int):
            pass
        elif isinstance(data[k], string_types):
            to_delete.append(k)
        elif isinstance(data[k], list):
            for s in data[k]:
                if isinstance(s, string_types):
                    to_delete.append(k)
                    break
            else:
                data[k] = to_variable(data[k])
        elif isinstance(data[k], torch.Tensor):
            data[k] = to_variable(data[k])
        else:
            assert False, "invalid tensorization of data dict"
    for k in to_delete:
        print("Deleting k=%s string data in dict" % k)
        del data[k]


def set_seed(seed, use_cuda):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)


def mk_module(mod, path="."):
    if isinstance(mod, str):
        mods = mod.split(".")
    else:
        assert isinstance(mod, list)
        mods = mod
    if len(mods) == 1:
        return
    curr_mod = mods[0]
    curr_path = join(path, curr_mod)
    if not os.path.exists(join(curr_path, "__init__.py")):
        os.system("touch %s" % join(curr_path, "__init__.py"))
    mk_module(mods[1:], path=curr_path)

"""
These two utilities are used to handle missing distributions / other pyro compatibility hacks
Change things in both pyro dists and stan code!
"""
def do_pyro_compatibility_hacks(code):
    # 1. replace all inv_gamma with normal distributions
    code = code.replace("inv_gamma", "normal")


def import_by_string(full_name):
    #try:
    module_name, unit_name = full_name.rsplit('.', 1)
    return getattr(__import__(module_name, fromlist=['']), unit_name)
    #except SyntaxError as e:
    #    raise (e)


def variablize_params(params):
    for k in params:
        params[k] = to_variable(params[k], requires_grad=True)


def to_variable(x, requires_grad=False):
    if isinstance(x, collections.Iterable):
        return torch.tensor(x, requires_grad=requires_grad)
    elif isinstance(x, torch.Tensor):
        return x
    return torch.tensor([x], requires_grad=requires_grad)


def handle_error(stage, e, etb=None):
    trace_v = log_traceback(e,etb)
    err_id = -1
    if stage == "run_transformed_data":
        if isinstance(e, KeyError):
            trace_v = "KeyError in transforming data: %s" % trace_v
            err_id = 8
        elif isinstance(e, AssertionError) and "Cannot handle function" in str(e):
            err_id = 12
    elif stage == "generate_pyro_file":
        if isinstance(e, AssertionError) and "SYNTAX ERROR in Stan Code" in str(e):
            err_id = 13
            if "const stan::lang::" in trace_v:
                err_id = 18

    elif stage == "json_file_to_mem_format":
        if isinstance(e, AssertionError) and "invalid json file data" in str(e):
            err_id = 15
    elif stage == "import_pyro_code":
        if (isinstance(e, SyntaxError) and "invalid syntax" in str(e)) or \
                (isinstance(e, AssertionError) and "model is None" in str(e)):
            err_id = 3
        elif isinstance(e, SyntaxError) and "can't assign to function call" in str(e):
            err_id = 3
        elif isinstance(e, AssertionError) and "init_params is None" in str(e):
            err_id = 4
    elif stage == "validate_data_def":
        err_id = 15
        trace_v = "original data validation failed: %s" % trace_v
    elif stage == "load_stan_code":
        if isinstance(e, AssertionError) and "increment_log_prob used in Stan code" in str(e):
            err_id = 7
    elif stage == "run_init_params":
        if "Cannot handle function" in str(e):
            err_id = 12
        elif "shape mismatch!" in str(e):
            err_id = 17
    elif stage == "run_stan":
        if "mismatch" in str(e) and "dimension" in str(e) and "declared and found in context" in str(e):
            err_id = 10
        elif "accessing element out of range" in str(e) or "Initialization failed" in str(e) \
                or "is neither int nor float nor list/array thereof" in str(e):
            err_id = 13

    elif stage == "run_pyro":
        if "dist_name=" in str(e) or "logits allowed in bernoulli, categorical only" in str(e):
            err_id = 11
        elif "Cannot handle function" in str(e) or "inhomogeneous total_count is not supported" in str(e):
            err_id = 12
        elif "Multiple pyro.sample sites named" in str(e) or "is not defined" in str(e) or \
                "tensors used as indices must be long or byte tensors" in str(e):
            err_id = 3
    elif stage == "log_prob_comparison":
        if "Log probs don't match" in str(e) and isinstance(e, AssertionError):
            err_id = 6
    assert err_id != -1, "cannot handle this error"
    return err_id, "[stage=%s] : %s" % (stage, trace_v)
