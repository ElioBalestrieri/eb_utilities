clear all
close all
clc


%% define parameters and create parameter space

[t_L, t_R, cfg] = funSpace;

%% define arbitrary params of simulated observer

obs1.alpha = -.1;
obs1.lambda = .01;
obs1.beta = 4;

obs2.alpha = .1;
obs2.lambda = .1;
obs2.beta = 6;



%% start looping

figure
summary_run = nan(cfg.ntrials, 3);

tic
for itrials = 1:cfg.ntrials
    
    % random select contrast value
    this = funEntropy(t_L, cfg);    
    summary_run(itrials,1) = this.x;
    
    % simulate 2 indep observers response
    which_obs = randsample(2, 1);
    if which_obs == 1
        this_obs = obs1;
    else
        this_obs = obs2;
    end
    this = funObserver(cfg, this_obs, this);
    summary_run(itrials,2) = which_obs;
    summary_run(itrials,3) = this.r;

    
    % update prior based on response
    t_L = funUpdate(t_L, this, cfg);
    
    % get parameter estimate
    this = funEstimate(t_L, this, cfg);
    
    % fancy plot
    subplot(2,3,1)
    plot(cfg.vals.cnts, cfg.FH.logistic(this.my_est, cfg.vals.cnts), 'k')
    hold on
    plot(cfg.vals.cnts, cfg.FH.logistic(obs1,  cfg.vals.cnts), 'b')
    plot(cfg.vals.cnts, cfg.FH.logistic(obs2,  cfg.vals.cnts), 'r')
    
    hold off
    legend('approx', 'true obs')
    title(['trialnum ' num2str(itrials)])
    
    subplot(2,3,2)
    plot(cfg.vals.alpha, this.alpha)
    title('alpha')
    
    subplot(2,3,3)
    plot(cfg.vals.beta, this.beta)
    title('beta')
    
%     subplot(2,3,4)
%     plot(cfg.vals.lambda, this.lambda)
%     title('lambda')
    
    subplot(2,3,5)
    plot(cfg.vals.cnts, this.info_gain)
    title('info gain')

    subplot(2,3,6)
    histogram(summary_run(:,1), cfg.npars.cnts)
    title('n choices')
    xlim(minmax(cfg.vals.cnts))
        
    pause(.010)
    
end
toc

%% now do the same, but after splitting
% hence without entropy calculation

[t_1, t_2, cfg] = funSpace;

figure

tic
for itrials = 1:cfg.ntrials
    
    this.x = summary_run(itrials,1);
    this.r = summary_run(itrials,3);
    which_obs = summary_run(itrials,2);
    
    % simulate 2 indep observers response
    if which_obs == 1

        % update prior based on response
        t_1 = funUpdate(t_1, this, cfg);
        
        % get parameter estimate
        this1 = funEstimate(t_1, this, cfg);

    else
        
        % update prior based on response
        t_2 = funUpdate(t_2, this, cfg);
        
        % get parameter estimate
        this2 = funEstimate(t_2, this, cfg);

    end
    
    
if itrials>10
    
    % fancy plot
    plot(cfg.vals.cnts, cfg.FH.logistic(this1.my_est, cfg.vals.cnts), 'b')
    hold on
    plot(cfg.vals.cnts, cfg.FH.logistic(this2.my_est, cfg.vals.cnts), 'r')
    plot(cfg.vals.cnts, cfg.FH.logistic(obs1,  cfg.vals.cnts), '--b')
    plot(cfg.vals.cnts, cfg.FH.logistic(obs2,  cfg.vals.cnts), '--r')
    hold off
    
    pause(.010)

end

end
toc





