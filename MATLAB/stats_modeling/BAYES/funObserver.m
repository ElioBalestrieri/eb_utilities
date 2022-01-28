function this = funObserver(cfg, obs, this)
% simulate observer response
% the function gives a binomial response (1 0, test vs standard) for the
% simulated observer with the pre-specified coefficents (alpha, beta, lambda)
% given the present contrast value

p = cfg.FH.logistic(obs, this.x);
this.r = randsample([1 0], 1, true, [p, 1-p]);



end