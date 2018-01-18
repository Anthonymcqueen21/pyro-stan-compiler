
# import some dependencies
import torch
from torch.autograd import Variable
import numpy as np
import json
import collections

def to_variable(x, requires_grad = False):
    if isinstance(x, collections.Iterable):
        return Variable(torch.FloatTensor(x), requires_grad=requires_grad)
    elif isinstance(x, torch.Tensor):
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor([x]), requires_grad=requires_grad)
    
E = 1e-5
def fudge(tp):
    return E + tp * (1-2*E)
    """
    p = tp.data[0]
    E = 1e-7
    assert (p >= 0 and p <= 1)
    if p < E:
        p = E
    elif p > 1-E:
        p = 1-E
    return to_variable(p)
    """
def load_data(fname):
    with open(fname,"r") as f:
        rdata = json.load(f)
    assert len(rdata) == 2
    data = {}
    n = len(rdata[0])
    for i in range(n):
        key = rdata[0][i]
        data[key] = rdata[1][i]
    return data

def do_analysis(model_name): 
    # TODO: implement model specific 
    pass
