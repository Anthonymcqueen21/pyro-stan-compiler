import errno
import os
import json
import pickle
import torch
import collections
from torch.autograd import Variable
import numpy as np
import pyro.distributions as pdist
from os.path import join
import pyro

def _call_func(fname, args):
    if fname.startswith("stan::math::"):
        fname=fname.split("stan::math::")[1]

    if len(args) == 1:
        x = args[0]
        if fname == "sqrt":
            return torch.sqrt(x)
    elif len(args) == 3:
        [x, y, z] = args
        if fname == "fma":
            return x*y+z
    else:
        torch_funmap = {
            "fmin" : "min",
            "fmax" : "max",
        }
        if fname in torch_funmap:
            fname = torch_funmap[fname]
        try:
            return getattr(torch,fname)(*args)
        except:
            assert False, "Cannot handle function=%s(%s)" % (fname,args)
def fma(x,y,z):
    return x*y+z

EPSILON = 1e-7

cache_init = {}


def sanitize_module_loading_file(pfile):
    if pfile.endswith(".py"):
        pfile = pfile[:-3]
    pfile = pfile.replace("/", ".")
    pfile = pfile.strip(".")
    return pfile

def generate_pyro_file(mfile, pfile):
    with open(pfile, "w") as f:
        f.write("from utils import to_variable, to_float, init_real_and_cache, _pyro_sample, _call_func\n")
        f.write("import torch\nimport pyro\n")

    os.system("../stan2pyro/bin/stan2pyro %s >> %s" % (mfile, pfile))

def get_fns_pyro(pfile):
    pfile = sanitize_module_loading_file(pfile)
    mk_module(pfile)
    model = import_by_string(pfile + ".model")
    assert model is not None, "model couldn't be imported"
    transformed_data = import_by_string(pfile + ".transformed_data")
    init_params = import_by_string(pfile + ".init_params")
    return init_params, model, transformed_data

def _pyro_sample(lhs, name, dist_name, dist_args, dist_kwargs=None,  obs=None):
    if dist_kwargs is None:
        dist_kwargs = {}
    reshaped_dist_args = []
    if dist_name.startswith("logit_"):
        dist_part = dist_name.split("_")[1]
        assert dist_part in ["bernoulli", "categorical"], "logits allowed in bernoulli and categorical only"
        dist_name = dist_part.capitalize()
        assert len(dist_args) == 1
        dist_kwargs["logit"] = dist_args[0]
        dist_args = []
    else:
        dist_name = dist_name.capitalize()

    try:
        dist_class = getattr(dist, dist_name)
    except:
        assert False, "dist_name=%s is invalid" % dist_name
    reshaped_dist_args = [arg.expand_as(lhs) for arg in dist_args]
    reshaped_dist_kwargs = {k: dist_kwargs[k].expand_as(lhs) for k in dist_kwargs}
    return pyro.sample(name, dist_class(*reshaped_dist_args, **reshaped_dist_kwargs), obs=obs)



def tensorize_data(data):
    for k in data:
        if isinstance(data[k], float) or isinstance(data[k], int):
            pass
        elif isinstance(data[k], list):
            data[k] = to_variable(data[k])
        elif isinstance(data[k], torch.Tensor):
            data[k] = to_variable(data[k])
        elif isinstance(data[k], Variable):
            pass
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
    cache_init = {}

def init_real_and_cache(name, low=None, high=None, dims=(1)):
    if name in cache_init:
        return cache_init[name]
    if low is None:
        low = -2.
        if high is not None and low >= high:
            low = high -1.
    if high is None:
        high = 2.
        if low >= high:
            high = low + 1.
    cache_init[name] = dist.Uniform(to_variable(low).expand(dims), to_variable(high).expand(dims))()
    return cache_init[name]

def import_by_string(full_name):
    try:
        module_name, unit_name = full_name.rsplit('.', 1)
        return getattr(__import__(module_name, fromlist=['']), unit_name)
    except:
        return None

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

def to_float(x):
    if isinstance(x, torch.Tensor):
        if len(x.shape) ==0:
            return float(x)
        assert len(x) == 1
        return x[0]
    elif isinstance(x, collections.Iterable):
        c = 0
        for val in x:
            c+=1
        assert c == 1
        for val in x:
            return val

    elif isinstance(x, Variable):
        return x.item()
    else:
        return float(x)

def variablize_params(params):
    for k in params:
        params[k] = to_variable(params[k], requires_grad=True)

def to_variable(x, requires_grad=False):
    if isinstance(x, torch.Tensor):
        return Variable(x, requires_grad=requires_grad)
    elif isinstance(x, collections.Iterable):
        return Variable(torch.FloatTensor(x), requires_grad=requires_grad)
    elif isinstance(x, torch.Tensor):
        return Variable(x, requires_grad=requires_grad)
    elif isinstance(x, Variable):
        return x
    else:
        return Variable(torch.FloatTensor([x]), requires_grad=requires_grad)

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