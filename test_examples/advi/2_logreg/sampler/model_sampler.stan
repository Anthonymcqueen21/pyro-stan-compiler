data{
    int<lower=0> N;
    int<lower=0> n_age;
    int<lower=0> n_age_edu;
    int<lower=0> n_edu;
    int<lower=0> n_region_full;
    int<lower=0> n_state;
    int<lower=0,upper=n_age> age[N];
    int<lower=0,upper=n_age_edu> age_edu[N];
    vector<lower=0,upper=1> [N]black;
    int<lower=0,upper=n_edu> edu[N];
    vector<lower=0,upper=1> [N]female;
    int<lower=0,upper=n_region_full> region_full[N];
    int<lower=0,upper=n_state> state[N];
    vector[N]v_prev_full;
}
parameters{
}
model{
}
generated quantities {
    vector[n_age]a;
    vector[n_edu]b;
    vector[n_age_edu]c;
    vector[n_state]d;
    vector[n_region_full]e;
    vector[5]beta;
    real<lower=0,upper=100> sigma_a=20;
    real<lower=0,upper=100> sigma_b=40;
    real<lower=0,upper=100> sigma_c=60;
    real<lower=0,upper=100> sigma_d=80;
    real<lower=0,upper=100> sigma_e=100;
    vector[N]y_hat;
    int<lower=0,upper=1> y[N];
    for(i in 1:n_age)
        a[i]=normal_rng(0,sigma_a);
    for(i in 1:n_edu)
        b[i]=normal_rng(0,sigma_b);
    for(i in 1:n_age_edu)
        c[i]=normal_rng(0,sigma_c);
    for(i in 1:n_state)
        d[i]=normal_rng(0,sigma_d);
    for(i in 1:n_region_full)
        e[i]=normal_rng(0,sigma_e);
    for(i in 1:5)
        beta[i]=normal_rng(0,100);
    for(i in 1:N){
        y_hat[i]=  beta[1]
                    +beta[2]*black[i]
                    +beta[3]*female[i]
                    +beta[5]*female[i]*black[i]
                    +beta[4]*v_prev_full[i]
                    +a[age[i]]
                    +b[edu[i]]
                    +c[age_edu[i]]
                    +d[state[i]]
                    +e[region_full[i]];
        y[i]=bernoulli_logit_rng(y_hat[i]);
    }
}