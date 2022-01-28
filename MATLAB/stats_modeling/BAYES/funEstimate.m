function this = funEstimate(T, this, cfg)

new_prior = T.prior;

% compute marginal posterior distributions
% dimord: alpha, beta, lambda
this.alpha = sum(new_prior,2);
this.beta = sum(new_prior,1)';
% this.lambda = squeeze(sum(sum(new_prior,2), 1)); % fixed

% compute coefficients trough weighted sum 
this.my_est.alpha = cfg.vals.alpha*this.alpha;
this.my_est.beta = cfg.vals.beta*this.beta;
% this.my_est.lambda = cfg.vals.lambda*this.lambda; % fixed

% compatibility mode
this.est = [this.my_est.alpha, this.my_est.beta]; %, this.my_est.lambda];

end