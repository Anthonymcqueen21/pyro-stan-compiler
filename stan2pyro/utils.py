
# import some dependencies
import torch
from torch.autograd import Variable
import numpy as np
from pdb import set_trace as bb
import json
import collections
import pyro
import pyro.distributions as dist

class SmoothedUniform(dist.Uniform):
    def __init__(self, *args, **kwargs):
        super(SmoothedUniform, self).__init__(*args, **kwargs)

    def batch_log_pdf(self, x):
        if len(x.size()) == 1 and x.size(0) == 1:
            v = x.data[0]
            if v < self.a.data[0] or v > self.b.data[0]:
                return Variable(torch.ones(1) * -100.)

        return super(SmoothedUniform, self).batch_log_pdf(x)

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
