import os
import os.path
from utils import mkdir_p
from divide_stan_data import divide_data

def get_all_data_paths(root, ofldr):
    iofiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith(".data.R"):

                dfile = os.path.join(path, name)
                nfldr = os.path.join(ofldr, (os.path.basename(path)))
                mkdir_p(nfldr)
                model_file = os.path.join(path, name.replace(".data.R", ".stan"))
                os.system("cp %s %s" % (model_file, nfldr))
                #nfile = os.path.join(nfldr, name)
                iofiles.append((dfile,nfldr))
    return iofiles

def convert_all_data(efldr, ofldr):
    iofiles = get_all_data_paths(efldr, ofldr)
    rate = (0,0)
    for dfile, nfldr in iofiles:
        print("working on %s" %dfile)
        jfile = os.path.join(nfldr, (".".join(os.path.basename(dfile).split(".")[:-1])) + ".json")
        os.system("Rscript --vanilla convert_data.R %s %s" % (dfile, jfile))
        if not os.path.exists(jfile):
            print("Error in Rscript")
            success=False
        else:
            success = divide_data(jfile, nfldr)
        rate = (rate[0]+success, rate[1]+1)
        print "%d/%d success" % (rate[0], rate[1])

    print "Finally %d/%d success" % (rate[0], rate[1])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--examples-folder', required=True, type=str, help="Examples Stan folder")
    parser.add_argument('-o', '--output-folder', required=True, type=str, help="Output folder for all data splits")
    args = parser.parse_args()
    convert_all_data(args.examples_folder, args.output_folder)