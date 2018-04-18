import pyro
import torch
import math
import collections
import math
import numpy as np
import pyro.distributions as dist
from .logger import log_traceback
from .compiler_utils import to_variable

cache_init = {}


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


def _call_func(fname, args):
    kwargs ={}
    if fname.startswith("stan::math::"):
        fname=fname.split("stan::math::")[1]

    if len(args) == 3:
        [x, y, z] = args
        if fname == "fma":
            # FIXME: tensor support
            return x*y+z

    if len(args) == 1:
        [x] = args
        x = to_variable(x)
        if fname == "log10":
            return torch.log(x) / math.log(10.)
        elif fname == "inv_logit":
            return torch.exp(x) / (1. + torch.exp(x))
        elif fname == "Phi":
            dims = x.shape
            return dist.Normal(torch.zeros(dims), torch.ones(dims)).cdf(x)

    torch_funmap = {
        "fmin" : "min",
        "fmax" : "max",
        "multiply" : "mul",
        "mul" : "mul",
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
        return getattr(torch, fname)(*args, **kwargs)
    except Exception as e:
        print(e)
        assert False, "Cannot handle function=%s(%s,%s)" % (fname,len(args),len(kwargs))


def identity(x):
    return x


def fma(x,y,z):
    return x*y+z


def _pyro_sample(lhs, name, dist_name, dist_args, dist_kwargs=None,  obs=None):
    if dist_kwargs is None:
        dist_kwargs = {}

    dist_args = [to_variable(v) for v in dist_args]
    dist_kwargs = {k: to_variable(dist_kwargs[k]) for k in dist_kwargs}
    if obs is not None:
        obs = to_variable(obs)

    mapped_names = {
        # "multi_normal" : "MultivariateNormal"
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
    except Exception as e:
        _, _ , etb = sys.exc_info()
        assert False, "issue with dist_name=%s : %s" % (dist_name, log_traceback(e,etb))

    if isinstance(lhs, float) or isinstance(lhs, int):
        lhs = torch.tensor([lhs])
    elif len(lhs.shape) == 0:
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
            low = high - 1.
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
    assert False, "not caching params anymore"
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


def as_bool(x):
    if isinstance(x, bool) or isinstance(x, int):
        return x >= 1
    elif isinstance(x, float):
        return as_bool(int(x))
    elif isinstance(x, torch.Tensor):
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
