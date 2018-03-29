import traceback
import os
from os.path import join
import json
import pickle
import numpy as np
import collections
import errno


def log_traceback(ex, ex_traceback=None):
    if ex_traceback is None:
        ex_traceback = ex.__traceback__
    tb_lines = [ line.rstrip('\n') for line in
                 traceback.format_exception(ex.__class__, ex, ex_traceback)]
    trace_v = ("Exception: " + "\n".join(tb_lines))
    print(trace_v)
    return trace_v


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
    assert validate_json(rdata), "invalid json file data"
    assert len(rdata) == 2, "invalid json file data"
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

