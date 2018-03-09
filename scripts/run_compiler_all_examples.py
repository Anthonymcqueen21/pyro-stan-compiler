import os
import os.path
from utils import mkdir_p
from divide_stan_data import divide_data
from pdb import set_trace as bb
from test_compare_models import test_generic

def get_all_data_paths(root, ofldr):
    args = []
    i=0
    for path, subdirs, files in os.walk(root):
        for name in files:
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

status_to_issue = {0 : "success", 1 : "R data conversion failed", 2: "Data division failed",
                   3: "model is None", 4: "init_params is None", 5: "transformed_data is None",
                   6: "Log probs failed", 7: "Model uses increment_log_prob",
                   8: "transformed_data failed to run", 9: "KeyError in init_params - likely mismatch in data/model files"}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--examples-folder', required=True, type=str, help="Examples Stan folder")
    args = parser.parse_args()
    ofldr = './test_compiler'
    mkdir_p(ofldr)
    args = list(sorted(get_all_data_paths(args.examples_folder,ofldr)))
    print("%d =total possible (R data, stan model) pairs with the same name" % len(args))

    status = {k : [] for k in range(len(status_to_issue))}
    for (dfile,mfile,pfile,model_cache) in args:
        n_samples =1
        this_try = test_generic(dfile,mfile,pfile,n_samples,model_cache)
        status[this_try].append((dfile,mfile,pfile,model_cache))
        print("pyro-file: %s" % pfile)
        for k in status:
            print("%s : %d" % (status_to_issue[k], len(status[k])))

    for k in status:
        print("%s : %d" %(status_to_issue[k], len(status[k])))
    bb()