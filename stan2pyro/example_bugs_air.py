from pdb import set_trace as bb

data = {"alpha" : 4.48,
        "beta" : 0.76,
       "sigma2" : 81.14,
       "J" : 3,
       "y" : [21, 20, 15],
       "n" : [48, 34, 21],
       "Z" : [10, 30, 50]}
def run_stan():
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
    fit = sm.sampling(data=data, iter=1000, chains=4)
    print(fit)


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

    def t(x):
        if isinstance(x, collections.Iterable):
            return Variable(torch.FloatTensor(x))
        elif isinstance(x, torch.Tensor):
            return Variable(x)
        else:
            return Variable(torch.FloatTensor([x]))


    num_epochs = 20

    #parameters
    mu_theta1 = Variable(torch.FloatTensor([0]), requires_grad=True)
    sigma_theta1 = Variable(torch.FloatTensor([1]), requires_grad=True)

    mu_theta2 = Variable(torch.FloatTensor([0]), requires_grad=True)
    sigma_theta2 = Variable(torch.FloatTensor([1]), requires_grad=True)

    mu_X = Variable(torch.zeros(data['J']), requires_grad=True)
    sigma_X = Variable(torch.ones(data['J']), requires_grad=True)


    def model(n, y):
        theta1 = pyro.sample("theta1", dist.normal, t(0), t(32))
        theta2 = pyro.sample("theta2", dist.normal, t(0), t(32))
        # FOR AUTOMATION: be mindful of scalar / vector multiplication semantics
        X = pyro.sample("X", dist.normal, (data['alpha']) + (data['beta']) * t(data['Z']),
                        (t(np.sqrt(data['sigma2']))).expand((data['J'])) )

        pyro.sample("y",dist.binomial, theta1.data[0] + theta2.data[0] * X, n, obs=y)

    def guide(n, y):
        # register params
        pyro.param("mu_theta1", mu_theta1)
        pyro.param("sigma_theta1", sigma_theta1)

        pyro.param("mu_theta2", mu_theta2)
        pyro.param("sigma_theta2", sigma_theta2)

        pyro.param("mu_X", mu_X)
        pyro.param("sigma_X", sigma_X)
        theta1 = pyro.sample("theta1", dist.normal, mu_theta1, sigma_theta1)
        theta2 = pyro.sample("theta2", dist.normal, mu_theta2, sigma_theta2)
        # FOR AUTOMATION: be mindful of scalar / vector multiplication semantics
        X = pyro.sample("X", dist.normal, mu_X, sigma_X)


    # setup the optimizer
    adam_params = {"lr": 0.05}
    optimizer = Adam(adam_params)

    loss = SVI(model, guide, optimizer, loss="ELBO")
    val = 0
    for epoch in range(num_epochs):
        #for i in range(data['J']):
        val = loss.step(t(data['n']), t(data['y']))
        print("epoch %d loss %0.2f" % (epoch+1, val))

    l=[mu_theta1, sigma_theta1, mu_theta2, sigma_theta2, mu_X, sigma_X]
    print(list(map(lambda x: x.data, l))

run_pyro()
run_stan()
