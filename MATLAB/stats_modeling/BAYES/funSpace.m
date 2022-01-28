function [t_L, t_R, cfg] = funSpace
% create params space for Bayesian adaptive procedure

%% function handles

% logistic function definition
% cfg.FH.logistic     = @(c, x)   c.lambda     + (1-2*c.lambda) ./ ...
%                                          (1+exp(-c.beta.*(x-c.alpha)));

% determine fixed value of lambda
fix_lambda = .02;

% logistic function variation on log 10
cfg.FH.logistic     = @(c, x)   fix_lambda  + (1-2*fix_lambda) ./ ...
                                         (1+10.^(-c.beta.*(x-c.alpha)));

                                     
% define probs handle over interval -for uniform distribution-
cfg.FH.probs = @(x) ones(1, numel(x))/numel(x);


%% initial parameter: num, range, actual values 

% numerosity
cfg.npars.alpha     = 41;
cfg.npars.beta      = 51;
cfg.npars.lambda    = 41;
cfg.npars.cnts      = 31;

% ranges
% the current ranges have been chosen by fitting the most suitable
% distribution based on empirical data (respectively, normal, inverse
% gaussian and half-normal for alpha, beta and lambda) and from the fitted
% distribution selected in this way the icdf was computed at .025 and .975,
% encompassing .95 of the overall dist
% cfg.ranges.alpha    = [-.2042     .2115];
% cfg.ranges.beta     = [4.1516   16.5972];
% cfg.ranges.lambda   = [0          .0257];
% cfg.ranges.cnts     = [-.45  .45];


% ranges 2
% same thing done before, but with different contrst range (in logn instead
% of log10)
cfg.ranges.alpha    = [-.3313     .3313];
cfg.ranges.beta     = [1.22     10.88];
% cfg.ranges.lambda   = [0     .15];
cfg.ranges.cnts     = [-.45     .45];


% pars values
cfg.vals.alpha      = linspace(cfg.ranges.alpha(1), cfg.ranges.alpha(2), cfg.npars.alpha);                  
cfg.vals.beta       = linspace(cfg.ranges.beta(1), cfg.ranges.beta(2), cfg.npars.beta);
% cfg.vals.lambda     = linspace(cfg.ranges.lambda(1), cfg.ranges.lambda(2), cfg.npars.lambda);
cfg.vals.cnts       = linspace(cfg.ranges.cnts(1), cfg.ranges.cnts(2), cfg.npars.cnts);

% some minor change: try to use logarithmically spaced values for beta and
% % lambda given the skewness of the theoretical distribution
% cfg.vals.beta       = exp(linspace(log(cfg.ranges.beta(1)), log(cfg.ranges.beta(2)), cfg.npars.beta));
% cfg.vals.lambda     = exp(linspace(log(cfg.ranges.lambda(1)), log(cfg.ranges.lambda(2)), cfg.npars.lambda));


%% compute P dists, for single params and overall
cfg.probs = structfun(cfg.FH.probs, cfg.vals, 'UniformOutput', 0);

% for the time being assume uniform distribution of all parameters. This
% could change once I have a proper look at the PMFs gathered in these
% months.

[t_L.prior, t_R.prior] = deal(ones(cfg.npars.alpha, cfg.npars.beta)/...
                            (cfg.npars.alpha*cfg.npars.beta));

% [t_L.prior, t_R.prior] = deal(ones(cfg.npars.alpha, cfg.npars.beta)/...
%                             (cfg.npars.alpha*cfg.npars.beta*cfg.npars.lambda));

                        
%% compute universal prob of resp test
cfg = local_univresp(cfg);                        

%% miscellaneous (n trial..)

cfg.ntrials = 280;



end

%% #################### LOCAL FUNCTIONS ######################
function cfg = local_univresp(cfg)

univ = nan(cfg.npars.alpha, cfg.npars.beta, cfg.npars.cnts);
linear_univ = nan(cfg.npars.alpha*cfg.npars.beta, ...
    cfg.npars.cnts);

acc = 0;

for ibeta = 1:cfg.npars.beta

    for ialpha = 1:cfg.npars.alpha

        % update accumulator
        acc = acc+1;

        % create the current coefficient combination
        this_c.alpha  = cfg.vals.alpha(ialpha);
        this_c.beta   = cfg.vals.beta(ibeta);                

        % compute prob response toward test stimulus, for this
        % parameter combination, for this stimulus contrast
        [linear_univ(acc,:), univ(ialpha, ibeta, :)]...
            = deal(cfg.FH.logistic(this_c, cfg.vals.cnts));

    end

end

    

cfg.univ = univ;
cfg.linear_univ = linear_univ;

end



