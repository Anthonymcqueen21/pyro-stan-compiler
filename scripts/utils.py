import errno
import os
import json
import pickle
import torch
import collections
import numpy as np
import pyro.distributions as pdist
from os.path import join
import pyro
import math
import subprocess
from six import string_types
from pdb import set_trace as bb

def _index_select(arr, ix):
    if isinstance(ix, int):
        return arr[ix]
    elif isinstance(ix,torch.Tensor):
        if len(ix.shape) == 0:
            assert float(ix) == int(ix), "ix should be interger"
            return arr[int(ix)]
        else:
            assert isinstance(arr, torch.Tensor)
            return torch.index_select(arr, 0, ix)
    else:
        assert False, "invalid index selection"

import traceback

def log_traceback(ex, ex_traceback=None):
    if ex_traceback is None:
        ex_traceback = ex.__traceback__
    tb_lines = [ line.rstrip('\n') for line in
                 traceback.format_exception(ex.__class__, ex, ex_traceback)]
    print("Exception: " + "\n".join(tb_lines))

def _call_func(fname, args):
    kwargs ={}
    if fname.startswith("stan::math::"):
        fname=fname.split("stan::math::")[1]

    if len(args) == 3:
        [x, y, z] = args
        if fname == "fma":
            return x*y+z

    if len(args) == 1:
        [x] = args
        if fname == "log10":
            return torch.log(x) / math.log(10.)

    torch_funmap = {
        "fmin" : "min",
        "fmax" : "max",
        "multiply" : "mul",
        "elt_multiply" : "mul",
        "subtract" : "sub",
        "fabs" : "abs",
        "sd" : "std",
        "divide" : "div",
        "elt_divide": "div",
        "logical_eq" : "eq",
    }

    if fname in torch_funmap:
        fname = torch_funmap[fname]
        if fname == "sd":
            kwargs["unbiased"] = False

    try:
        args = list(map(lambda x: to_variable(x), args))
        return getattr(torch,fname)(*args, **kwargs)
    except:
        assert False, "Cannot handle function=%s(%s,%s)" % (fname,args,kwargs)

def identity(x):
    return x

def fma(x,y,z):
    return x*y+z

EPSILON = 1e-7

cache_init = {}

def as_bool(x):
    if isinstance(x, bool) or isinstance(x, int):
        return x >= 1
    elif isinstance(x, float):
        return as_bool(int(x))
    elif isinstance(x, torch.Tenspor):
        assert len(x) == 1, "one_element allowed for Variable in as_bool"
        return as_bool(x.item())
    elif isinstance(x, collections.Iterable):
        ctr = 0
        v = None
        for v_ in x:
            v = v_
            ctr +=1
        assert (ctr == 1), "one_element allowed for Variable in as_bool"
        return as_bool(v)
    assert False, "Invalid type inside as_bool"

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
        f.write("from utils import to_float, init_real_and_cache, _pyro_sample, _call_func, check_constraints\n")
        f.write("from utils import init_real, init_vector, init_matrix, init_int, init_int_and_cache\n")
        f.write("from utils import init_vector_and_cache, _index_select, init_matrix_and_cache, to_int, _pyro_assign, as_bool\n")
        f.write("import torch\nimport pyro\n")
        f.write("from utils import identity as to_variable\n\n")
        f.write(out + "\n")
        for err_s in ["SYNTAX ERROR, MESSAGE(S) FROM PARSER", "Aborted (core dumped)", "SAMPLING CONSTANTS NOT SUPPORTED"]:
            assert err_s not in out and err_s not in err, "SYNTAX ERROR in Stan Code: %s" % err

    #os.system("../stan2pyro/bin/stan2pyro %s >> %s" % (mfile, pfile))

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

def _pyro_sample(lhs, name, dist_name, dist_args, dist_kwargs=None,  obs=None):
    if dist_kwargs is None:
        dist_kwargs = {}

    dist_args = [to_variable(v) for v in dist_args]
    dist_kwargs = {k: to_variable(dist_kwargs[k]) for k in dist_kwargs}
    if obs is not None:
        obs = to_variable(obs)

    mapped_names = {
        "Multi_normal" : "MultivariateNormal"
    }
    if dist_name.endswith("_logit"):
        dist_part = dist_name.split("_")[0]
        assert dist_part in ["bernoulli", "categorical"], "logits allowed in bernoulli, categorical only"
        dist_name = dist_part.capitalize()
        assert len(dist_args) == 1
        dist_kwargs["logits"] = dist_args[0]
        dist_args = []
    elif dist_name in mapped_names:
        dist_name = mapped_names[dist_name]
    else:
        dist_name = dist_name.capitalize()

    try:
        dist_class = getattr(dist, dist_name)
    except:
        assert False, "dist_name=%s is invalid" % dist_name

    if len(lhs.shape) == 0:
        lhs = lhs.expand((1))
    reshaped_dist_args = [arg.expand_as(lhs) for arg in dist_args]
    reshaped_dist_kwargs = {k: dist_kwargs[k].expand_as(lhs) for k in dist_kwargs}

    return pyro.sample(name, dist_class(*reshaped_dist_args, **reshaped_dist_kwargs), obs=obs)

def _pyro_assign(lhs, rhs):
    if isinstance(lhs, torch.Tensor) or isinstance(lhs, torch.Tensor):
        shape_dim = len(lhs.shape)
        if shape_dim == 0 or (shape_dim == 1 and lhs.shape[0]==1):
            return to_float(rhs)
        else:
            return rhs.expand_as(lhs)
    elif isinstance(lhs,float):
        return to_float(rhs)
    elif isinstance(lhs,int):
        return to_int(rhs)
    else:
        assert False, "invalid lhs type: %s" % (lhs)

def tensorize_data(data):
    for k in data:
        if isinstance(data[k], float) or isinstance(data[k], int):
            pass
        elif isinstance(data[k], list):
            data[k] = to_variable(data[k])
        elif isinstance(data[k], torch.Tensor):
            data[k] = to_variable(data[k])
        else:
            assert False, "invalid tensorization of data dict"

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
    return code

class DIST(dict):
    def __getattr__(self, x):
        try:
            return getattr(pdist, x)
        except:
            if x == "Inv_gamma":
                return pdist.Normal
            else:
                raise
dist = DIST()

def reset_initialization_cache():
    global cache_init
    cache_init = {}

def init_matrix_and_cache(name,  low=None, high=None, dims=None):
    assert dims is not None, "dims cannot be empty for a matrix"
    return init_real_and_cache(name, low=low, high=high, dims=dims)


def init_vector_and_cache(name,  low=None, high=None, dims=None):
    assert dims is not None, "dims cannot be empty for a vector"
    return init_real_and_cache(name, low=low, high=high, dims=dims)


def init_matrix(name,  low=None, high=None, dims=None):
    assert dims is not None, "dims cannot be empty for a vector"
    return init_real(name, low=low, high=high, dims=dims)

def init_vector(name,  low=None, high=None, dims=None):
    assert dims is not None, "dims cannot be empty for a vector"
    return init_real(name, low=low, high=high, dims=dims)

def init_real(name, low=None, high=None, dims=(1)):
    if isinstance(dims, float) or isinstance(dims, int):
        dims = [to_int(dims)]
    if low is None:
        low = -2.
        if high is not None and low >= high:
            low = high -1.
    if high is None:
        high = 2.
        if low >= high:
            high = low + 1.
    r = dist.Uniform(to_variable(low).expand(dims), to_variable(high).expand(dims)).sample()
    assert r is not None
    return r

def init_int(name, low=None, high=None, dims=(1)):
    if isinstance(dims, float) or isinstance(dims, int):
        dims = [to_int(dims)]
    if dims == [1]:
        r = 0
    else:
        r = torch.zeros(dims)
    return r

def init_int_and_cache(name, low=None, high=None, dims=(1)):
    if isinstance(dims, float) or isinstance(dims, int):
        dims = [to_int(dims)]
    if name in cache_init:
        assert cache_init[name] is not None
        dims_lst = [dims] if isinstance(dims, int) else list(dims)
        assert len(dims_lst) == len(cache_init[name].shape)
        for i in range(len(dims_lst)):
            assert cache_init[name].shape[i] == dims_lst[i], "shape mismatch!"
        return cache_init[name]
    cache_init[name] = init_int(name,low=low,high=high,dims=dims)
    return cache_init[name]

def init_real_and_cache(name, low=None, high=None, dims=(1)):
    if isinstance(dims, float) or isinstance(dims, int):
        dims = [to_int(dims)]
    if name in cache_init:
        assert cache_init[name] is not None
        dims_lst = [dims] if isinstance(dims, int) else list(dims)
        assert len(dims_lst) == len(cache_init[name].shape)
        for i in range(len(dims_lst)):
            assert cache_init[name].shape[i] == dims_lst[i], "shape mismatch!"
        return cache_init[name]
    cache_init[name] = init_real(name,low=low,high=high,dims=dims)
    return cache_init[name]

def check_constraints(v, low=None, high=None, dims=None):
    if dims == []:
        dims=[1]
    assert dims is not None, "dims must be specified in check_constraints"
    def check_l_h_float(v_):
        assert low is None or v_ >= low, "low constraint not satsfied, v=%s low=%s" % (v_, low)
        assert high is None or v_ <= high, "high constraint not satsfied, v=%s high=%s" % (v_, high)

    if isinstance(v, int) or isinstance(v, float):
        check_l_h_float(v)
        assert dims == [1], "dims of int/float mismatched v=%s,dims=%s" % (v, dims)
    elif isinstance(v, list):
        n_ = len(v)
        assert len(dims) >= 1, "invalid dims; expected=%d, found: None" % (n_)
        assert n_ == dims[0], "dimension mismatch expected=%d, found=%d" % (n_, dims[0])
        for v_i in v:
            check_constraints(v_i, low=low, high=high, dims=dims[1:])
    else:
        assert False, "invalid data type for v=%s" % v


def import_by_string(full_name):
    #try:
    module_name, unit_name = full_name.rsplit('.', 1)
    return getattr(__import__(module_name, fromlist=['']), unit_name)
    #except SyntaxError as e:
    #    raise (e)

def save_p(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_p(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

def exists_p(fname):
    return os.path.exists(fname)



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def to_int(x):
    fx = to_float(x)
    assert int(fx) == fx, "value was a float but not int!"
    return int(fx)

def to_float(x):
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 0:
            return float(x)
        assert len(x) == 1
        return x[0]
    elif isinstance(x, collections.Iterable):
        c = 0
        for val in x:
            c += 1
        assert c == 1
        for val in x:
            return val
    else:
        return float(x)

def variablize_params(params):
    for k in params:
        params[k] = to_variable(params[k], requires_grad=True)

def to_variable(x, requires_grad=False):
    if isinstance(x, torch.Tensor):
        return torch.tensor(x, requires_grad=requires_grad)
    elif isinstance(x, collections.Iterable):
        return torch.tensor(x, requires_grad=requires_grad)
    elif isinstance(x, torch.Tensor):
        return x
    return torch.tensor([x], requires_grad=requires_grad)

def validate_json(rdata):
    n = len(rdata[0])
    if len(rdata[1]) != n:
        return False
    for i in range(n):
        if isinstance(rdata[1][i], collections.Iterable):
            if not isinstance(rdata[1][i], list):
                return False
            try:
                np.array(rdata[1][i])
            except:
                return False
    return True


def json_file_to_mem_format(rdata):
    assert validate_json(rdata)
    assert len(rdata) == 2
    data = {}
    n = len(rdata[0])
    for i in range(n):
        key = rdata[0][i]
        if key != "args":
            data[key] = rdata[1][i]
    return data

def load_data(fname):
    with open(fname,"r") as f:
        rdata = json.load(f)
    return json_file_to_mem_format(rdata)
