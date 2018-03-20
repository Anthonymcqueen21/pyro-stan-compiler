import os
import os.path
from utils import mkdir_p
from divide_stan_data import divide_data
from pdb import set_trace as bb
from test_compare_models import test_generic, cache_all_models

def get_all_data_paths(root, ofldr):
    args = []
    i=0
    for path, subdirs, files in os.walk(root):
        for name in sorted(files):
            if name.endswith(".data.R"):

                dfile = os.path.join(path, name)
                pre_name =  name.split(".data.R")[0]
                mfile = os.path.join(path,pre_name + ".stan")
                if os.path.exists(mfile):
                    i += 1
                    pfile = os.path.join(ofldr, "auto_%d_%s.py" % (i, pre_name.replace(".","_")))
                    model_cache = os.path.join(ofldr, "mc_auto_%d_%s.pkl" % (i, pre_name.replace(".","_")))
                    args.append((dfile,mfile,pfile,model_cache))
    return args

status_to_issue = {0 : "success",
                   1 : "R data conversion failed",
                   2: "Data division failed",
                   3: "model is None / Python Syntax Error / Def Incorrect Pyro code",
                   4: "init_params is None",
                   5: "transformed_data is None",
                   6: "Log probs failed",
                   7: "Model uses increment_log_prob",
                   8: "transformed_data failed to run",
                   9: "KeyError in init_params - likely mismatch in data/model files",
                   10: "StanRuntimeError: mismatch in dimensions declared and found in context",
                   11: "distribution not implemented in pyro / not connected with Stan / issue with logits",
                   12: "Cannot handle function",
                   13: "StanRuntimeError accessing element out of range / Initialization failed / Syntax Error in Stan",
                   14: "variable values in data.R file are nested dictionaries!",
                   15: "original data validation failed",
                   16: "splitted data validation failed",
                   17: "param shaped are probably dependent on data (shape mismatch in cache!)",
                   18: "Feature not implemented in Stan2Pyro.cpp"}

import pickle
def get_cached_state(fname):
    with open(fname, "rb") as f:
        [j, status] = pickle.load(f)
    return (j,status)

def save_cached_state(j, status, fname):
    with open(fname, "wb") as f:
        pickle.dump([j,status], f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--examples-folder', required=True, type=str, help="Examples Stan folder")
    parser.add_argument('-i', '--eid', default=None, type=int, help="example id to run")
    p_args = parser.parse_args()
    ofldr = './test_compiler'
    mkdir_p(ofldr)
    args = list(sorted(get_all_data_paths(p_args .examples_folder,ofldr)))
    print("%d =total possible (R data, stan model) pairs with the same name" % len(args))

    if p_args.eid is not None:
        import sys

        for (dfile, mfile, pfile, model_cache) in args:
            if ("_%s_" % p_args.eid) in pfile:
                break
        
        n_runs = 2
        this_try = test_generic(dfile, mfile, pfile, n_runs, model_cache)
        print(status_to_issue[this_try])
        print(args[p_args.eid])
        bb()
        sys.exit(0)
    status = {k : [] for k in range(len(status_to_issue))}
    j=0
    cfname = "%s/status.pkl" % ofldr
    if os.path.exists(cfname):
        (j,status) = get_cached_state(cfname)
        args = args[j+1:]
        j=j+1
        for  k in range(len(status_to_issue)):
            if k not in status:
                status[k] = []

    #cache_all_models(args)
    #bb()

    for (dfile,mfile,pfile,model_cache) in args:
        n_runs = 2
        print("STARTING TO PROCESS %d: pyro-file: %s" % (j, pfile))
        this_try, err = test_generic(dfile,mfile,pfile,n_runs,model_cache)
        #if err is not None and "const" in err and "Assertion" in err and "false" in err:
        #    bb()
        status[this_try].append((dfile,mfile,pfile,model_cache,err))

        for k in status:
            if len(status[k]) > 0:
                print("[%s]%s : %d" % (k, status_to_issue[k], len(status[k])))
        if cfname is not None:
            save_cached_state(j, status, cfname)
        j+=1
    for k in status:
        if len(status[k]) > 0:
            print("[%s] %s : %d" %(k, status_to_issue[k], len(status[k])))
    bb()