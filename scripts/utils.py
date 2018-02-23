import errno
import os
import json
import pickle
import torch
import collections
from torch.autograd import Variable
import numpy as np

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

def to_variable(x, requires_grad=False):
    if isinstance(x, collections.Iterable):
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


def load_data(fname):
    with open(fname,"r") as f:
        rdata = json.load(f)
    assert validate_json(rdata)
    assert len(rdata) == 2
    data = {}
    n = len(rdata[0])
    for i in range(n):
        key = rdata[0][i]
        if key != "args":
            data[key] = rdata[1][i]
    return data