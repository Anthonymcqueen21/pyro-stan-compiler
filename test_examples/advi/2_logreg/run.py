from pdb import set_trace as bb 
import pystan

def run_stan_advi(fname, data, iters=1000):
    with open(fname,"r") as f:
        code = f.read()
        
    sm = pystan.StanModel(model_code=code)
    fit = sm.vb(data=data, iter=iters)
    print(fit)
    bb()
    
    
run_stan_advi("model.stan")
