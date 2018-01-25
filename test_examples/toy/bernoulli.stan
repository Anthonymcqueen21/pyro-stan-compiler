data {
int<lower=0> N;
int<lower=0,upper=1> y[N];
real<lower=0,upper=1> theta;
}
parameters {
}
model {
}
generated quantities {
int<lower=0> sigma = 1;
real<lower=0,upper=1> mu;
real<lower=0,upper=1> z;
mu = bernoulli_rng(theta);
z=normal_rng(mu, sigma);
}
