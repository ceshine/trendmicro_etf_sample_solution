data {
  int<lower=0> N;
  vector[N] t; // Target
  vector[N] x_1; // Ret Minus 1
  vector[N] x_2; // Ret Minus 2
  vector[N] x_3; // Ret Minus 3
  vector[N] x_4; // Ret Minus 4
  vector[N] x_5; // Ret Minus 5
  real df; // Degrees of freedom (2.6)
}

parameters {
  real w_1_0;
  real w_1_1;
  real w_1_2;
  real w_1_3;
  real w_1_4;
  real w_1_5;
  real<lower=0> alpha;
  real<lower=0> beta;
}

transformed parameters {
  real<lower=0> sigma_w;
  real<lower=0> sigma_t;
  sigma_w = sqrt(alpha);
  sigma_t = sqrt(beta);
}

model {
  w_1_0 ~ normal(0,sigma_w);
  w_1_1 ~ normal(0,sigma_w);
  w_1_2 ~ normal(0,sigma_w);
  w_1_3 ~ normal(0,sigma_w);
  w_1_4 ~ normal(0,sigma_w);
  w_1_5 ~ normal(0,sigma_w);
  alpha ~ inv_gamma(1E-2, 1E-4);
  beta ~ inv_gamma(.3, .0001);
  t ~ student_t(
      df, w_1_0 + w_1_1*x_1 + w_1_2*x_2 +
      w_1_3*x_3 + w_1_4*x_4 + w_1_5*x_5, sigma_t);
}
