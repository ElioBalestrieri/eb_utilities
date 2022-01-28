function this = funEntropy(T, cfg)
%% the function first computes entropy for each soa and each possible
% behavioural response (yes vs no). Then compute the entropy of the current
% at the current trial, compute the difference between the expected
% entropies and the current (information gain). The expected information
% gain is obtained by a weghted sum of the matrix obtained before and the
% expected behavioural response for each soa based on the current prior.
% The SOA maximizing the information gain (that is, minimizing entropy) is
% chosen for the subsequent trial.


n_cnts = cfg.npars.cnts;
H_mat = nan(n_cnts, 2);

% start compute posteriors for each SOA, for each possible response
for icnt = 1:n_cnts
    
    for iresp = [0 1]
        
        % compute for r ==1;
        nextthis.r = iresp;
        nextthis.x = cfg.vals.cnts(icnt);

        % call to qUpdate
        out = funUpdate(T, nextthis, cfg);
        % compute entropy (but not sum yet)
        swap_mat = out.prior.*log(out.prior);
        % correction for lim xlogx as x->0 == 0
        swap_mat(isnan(swap_mat)) = 0;
        % put in the right pos, r=1 1st col, r=0 2nd col
        H_mat(icnt,abs(iresp-2)) = -sum(sum(swap_mat));

    end

    
end

% compute entropy of the current prior -same passages as before-
prior_H = T.prior.*log(T.prior);
prior_H(isnan(prior_H))=0;
prior_H = -sum(sum(prior_H));

% information gain as change between current prior and posterior
H_mat = prior_H-H_mat;
% linearize prior to facilitate computing
linear_prior = T.prior(:);
% compute probability of response based on the current prior
r1 = sum(cfg.linear_univ.*repmat(linear_prior, 1, n_cnts))';
% weights matrix of response (correct vs wrong)
weight_r = [r1, 1-r1];
% expected information gain
info_gain = sum(H_mat.*weight_r,2);
% find max
this.cnt_idx = find(info_gain==max(info_gain));
if numel(this.cnt_idx)>1
    warning('at least two points with same, max info gain')   
    this.cnt_idx = randsample(this.cnt_idx, 1);
end

% define contrast value to be displayed in the next trial
this.x = cfg.vals.cnts(this.cnt_idx);

% information gain
this.info_gain = info_gain;


end

