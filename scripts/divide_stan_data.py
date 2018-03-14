import copy
import json
import collections
import numpy as np
import os.path
import os
from utils import validate_json
from pdb import set_trace as bb
from six import string_types

def load_json(fname):
    with open(fname,"r") as f:
        j = json.load(f)
    return j

def save_json(fname, j):
    with open(fname,"w") as f:
        json.dump(j, f)

def is_int(val):
    if type(val) == int:
        return val
    elif type(val) == float:
        return int(val) == val
    else:
        return False

def get_arrays_to_divide(n, jdata, val):
    # see if breaking along this dimension (i, val) is okay
    # there should at least be one array that has this as its first dimension / len
    # no array should have any other first dimension
    # NOTE: this is not a perfect check -- if two variables have the same values, we can be fooled into
    #       splitting by one when we should be using the other -- solution should look at data types

    to_divide_ixs = []
    for j in range(n):
        if isinstance(jdata[1][j], collections.Iterable):
            dim1 = len(jdata[1][j])
            assert not isinstance(jdata[1][j], dict), "variable values in data.R file are nested dictionaries!"
            if isinstance(jdata[1][j][0], collections.Iterable):
                dim2 = len(jdata[1][j][0])
                if dim2 == dim1:
                    print("[N][N] shapes are not handled")
                    to_divide_ixs = None
                    break
                if isinstance(jdata[1][j][0][0], collections.Iterable):
                    print("3-dimensional shapes are not handled")
                    to_divide_ixs = None
                    break
            if dim1 == val:
                to_divide_ixs.append(j)
    if to_divide_ixs is None or len(to_divide_ixs) == 0:
        return None
    return to_divide_ixs

def print_shape(jdata):
    n = len(jdata[0])
    dims = []
    for i in range(n):
        if isinstance(jdata[1][i], collections.Iterable):
            dims.append(np.array(jdata[1][i]).shape)
        else:
            dims.append(1)
    print(dims)

def write_to_folder(ofolder, jd1, jd2, dfile):
    basename = os.path.basename(dfile)
    fname1 = os.path.join(ofolder, "a_" + basename)
    fname2 = os.path.join(ofolder, "b_" + basename)

    save_json(fname1, jd1)
    save_json(fname2, jd2)
    print("saved: fname1=%s fname2=%s " % (fname1, fname2))
    print_shape(jd1)
    print_shape(jd2)



def divide_json_data(jdata):
    if isinstance(jdata[0], string_types):
        jdata[0] = [jdata[0]]
    n = len(jdata[0])
    assert len(jdata) == 2 and len(jdata[1]) == n

    # identify all dims
    int_ixs, int_vals = [], []

    for i in range(n):
        var = jdata[0][i]
        val = jdata[1][i]
        if is_int(val) and val >= 2:
            to_divide_ixs = get_arrays_to_divide(n, jdata, val)
            if to_divide_ixs is not None:
                jd1 = copy.deepcopy(jdata)
                jd2 = copy.deepcopy(jdata)
                jd1[1][i] = int(val / 2)
                jd2[1][i] = val - int(val / 2)
                for j in to_divide_ixs:
                    jd1[1][j] = jd1[1][j][:int(val / 2)]
                    jd2[1][j] = jd2[1][j][int(val / 2):]

                return True, jd1, jd2
    print("Cannot divide this dataset! :(")
    print_shape(jdata)
    return False, None, None

def divide_data(dfile, ofolder):
    jdata = load_json(dfile)
    if not validate_json(jdata):
        print("Invalid JSON data: ignoring %s" % dfile)
        return False

    n = len(jdata[0])
    assert len(jdata) == 2 and len(jdata[1]) == n

    #identify all dims
    int_ixs, int_vals = [], []

    for i in range(n):
        var = jdata[0][i]
        val = jdata[1][i]
        if is_int(val) and val >= 2:
            to_divide_ixs = get_arrays_to_divide(n, jdata, val)
            if to_divide_ixs is not None:
                jd1 = copy.deepcopy(jdata)
                jd2 = copy.deepcopy(jdata)
                jd1[1][i] = int(val/2)
                jd2[1][i] = val - int(val / 2)
                for j in to_divide_ixs:
                    jd1[1][j] = jd1[1][j][:int(val/2)]
                    jd2[1][j] = jd2[1][j][int(val / 2):]
                write_to_folder(ofolder, jd1, jd2, dfile)
                return True
    print("Cannot divide this dataset! :(")
    print_shape(jdata)
    return False
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Divide Stan data into multiple (2) parts!")
    parser.add_argument('-d', '--data-file', required=True, type=str, help="JSON data file")
    parser.add_argument('-o', '--output-folder', required=True, type=str, help="Output folder")
    args = parser.parse_args()
    divide_data(args.data_file, args.output_folder)
