from pdb import set_trace as bb

data = {"alpha" : 4.48,
        "beta" : 0.76,
       "sigma2" : 81.14,
       "J" : 3,
       "y" : [21, 20, 15],
       "n" : [48, 34, 21],
       "Z" : [10, 30, 50]}
       
#data['sigma'] = np.sqrt(data['sigma2'])

def run_stan(iters, advi = False):
    import pystan

    code = """
    data {
      real alpha; 
      real beta; 
      real<lower=0> sigma2; 
      int<lower=0> J; 
      int y[J]; 
      vector[J] Z;
      int n[J]; 
    } 

    transformed data {
      real<lower=0> sigma; 
      sigma <- sqrt(sigma2); 
    } 

    parameters {
       real theta1; 
       real theta2; 
       vector[J] X; 
    } 

    model {
      real p[J];
      theta1 ~ normal(0, 32);   // 32^2 = 1024 
      theta2 ~ normal(0, 32); 
      X ~ normal(alpha + beta * Z, sigma);
      y ~ binomial_logit(n, theta1 + theta2 * X);
    }

    """


    sm = pystan.StanModel(model_code=code)
    if advi:
        fit = sm.vb(data=data, iter=iters)
    else:
        fit = sm.sampling(data=data, iter=iters, chains=4)
    return sm, fit


def run_pyro():
    # import some dependencies
    import torch
    from torch.autograd import Variable
    import torch.nn as nn
    import numpy as np
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI
    from pyro.optim import Adam

    import collections

    def t(x, requires_grad = False):
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
        return t(p)


    num_epochs = 1000

    def model(n, y, Z):
        theta1 = pyro.sample("theta1", dist.normal, t(0), t(32))
        theta2 = pyro.sample("theta2", dist.normal, t(0), t(32))
        
        for i in range(data['J']):
            # FOR AUTOMATION: be mindful of scalar / vector multiplication semantics
            X = pyro.sample("X[%d]" % i, dist.normal, (data['alpha']) + (data['beta']) * Z[i],
                            t(np.sqrt(data['sigma2'])) )
            logit_p = (theta1.data[0] + theta2.data[0] * X)
            pyro.sample("y[%d]" % i,dist.binomial, fudge(1.0/( torch.exp(-logit_p) + 1)) , n[i], obs=y[i])

    
    def guide(n, y, Z):
        
        #parameters -- initialization -- same as model?
        mu_theta1 = t(0, requires_grad=True)
        sigma_theta1 = t(32, requires_grad=True)

        mu_theta2 = t(0, requires_grad=True)
        sigma_theta2 = t(32, requires_grad=True)
      
        mu_X =  [t((data['alpha']) + (data['beta']) * data['Z'][i], requires_grad=True) for i in range(data['J'])]
        sigma_X = [t(np.sqrt(data['sigma2']), requires_grad=True) for i in range(data['J'])]

        # register params
        pyro.param("mu_theta1", mu_theta1)
        pyro.param("sigma_theta1", sigma_theta1)

        pyro.param("mu_theta2", mu_theta2)
        pyro.param("sigma_theta2", sigma_theta2)

        
        theta1 = pyro.sample("theta1", dist.normal, mu_theta1, sigma_theta1)
        theta2 = pyro.sample("theta2", dist.normal, mu_theta2, sigma_theta2)
        # FOR AUTOMATION: be mindful of scalar / vector multiplication semantics
        for i in range(data['J']):
            pyro.param("mu_X[%d]" % i, mu_X[i])
            pyro.param("sigma_X[%d]" %i, sigma_X[i])
            X = pyro.sample("X[%d]"%i, dist.normal, mu_X[i], sigma_X[i])


    # setup the optimizer
    adam_params = {"lr": 0.01}
    optimizer = Adam(adam_params)

    loss = SVI(model, guide, optimizer, loss="ELBO")
    val = 0
    min_loss = 10000000
    for epoch in range(num_epochs):
        #for i in range(data['J']):
        val = loss.step(t(data['n']), t(data['y']), t(data['Z']))
        if val <min_loss:
            min_loss = val
            print("epoch %d loss %0.2f" % (epoch+1, val))
            print("theta1 %0.3f %0.3f" % (mu_theta1.data[0], sigma_theta1.data[0]))
            print("theta2 %0.3f %0.3f" % (mu_theta2.data[0], sigma_theta2.data[0]))
            for i in range(data['J']):
                print("X[%d] %0.3f %0.3f" % (i, mu_X[i].data[0], sigma_X[i].data[0]))

        if epoch %100 == 1 and False:
            print("epoch %d loss %0.2f" % (epoch+1, val))

            print("theta1 %0.3f %0.3f" % (mu_theta1.data[0], sigma_theta1.data[0]))
            print("theta2 %0.3f %0.3f" % (mu_theta2.data[0], sigma_theta2.data[0]))
            for i in range(data['J']):
                print("X[%d] %0.3f %0.3f" % (i, mu_X[i].data[0], sigma_X[i].data[0]))
    

run_pyro()
"""
Sampling result:
         mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
theta1  -0.68    0.27   2.26  -3.56  -1.21  -0.65  -0.25    1.2   72.0   1.04
theta2   0.04    0.01   0.09  -0.03   0.02   0.04   0.06   0.14   64.0   1.05
X[0]     13.0    0.35   8.63   -4.1   7.34  12.96  18.66  29.99  608.0    1.0
X[1]    27.24    0.26   7.49  13.17  21.97  27.21  32.39  42.59  812.0    1.0
X[2]    40.84    0.46   8.64  23.96  34.81  40.75  46.59  57.49  359.0   1.01
lp__   -71.11    0.09   1.65  -75.1 -71.94 -70.76 -69.87 -68.97  347.0   1.01

"""
#STANOUT="Inference for Stan model: anon_model_ab6c28ff14c1f5084cf37a0a56b47c5a.\n4 chains, each with iter=1000; warmup=500; thin=1;\npost-warmup draws per chain=500, total post-warmup draws=2000.\n\n         mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat\ntheta1  -0.68    0.27   2.26  -3.56  -1.21  -0.65  -0.25    1.2   72.0   1.04\ntheta2   0.04    0.01   0.09  -0.03   0.02   0.04   0.06   0.14   64.0   1.05\nX[0]     13.0    0.35   8.63   -4.1   7.34  12.96  18.66  29.99  608.0    1.0\nX[1]    27.24    0.26   7.49  13.17  21.97  27.21  32.39  42.59  812.0    1.0\nX[2]    40.84    0.46   8.64  23.96  34.81  40.75  46.59  57.49  359.0   1.01\nlp__   -71.11    0.09   1.65  -75.1 -71.94 -70.76 -69.87 -68.97  347.0   1.01"
#print(STANOUT)

sm, fit = run_stan(iters=100000, advi=True)
bb()
