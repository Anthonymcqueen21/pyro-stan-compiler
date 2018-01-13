
# import some dependencies
import torch
from torch.autograd import Variable
import numpy as np

 def to_variable(x, requires_grad = False):
    if isinstance(x, collections.Iterable):
        return Variable(torch.FloatTensor(x), requires_grad=requires_grad)
    elif isinstance(x, torch.Tensor):
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor([x]), requires_grad=requires_grad)
    

def fudge(tp):
    p = tp.data[0]
    E = 1e-7
    assert (p >= 0 and p <= 1)
    if p < E:
        p = E
    elif p > 1-E:
        p = 1-E
    return to_variable(p)
    
def load_data(model_name): 
    # for now, load in memory
    # TODO: make more efficient
    data = {}
    return data

