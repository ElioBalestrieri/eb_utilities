function T = funUpdate(T, this, cfg)
%% T = funUpdate(T, this, cfg)
%  updates prior on the basis of the current response according to Bayes
%  theorem.

%% get params
r = this.r;
prior = T.prior;
idx_cnt = find(cfg.vals.cnts==this.x);

univ = cfg.univ;

%% apply Bayes theorem
T.prior = local_Pt_given_parsANDx(univ, r, idx_cnt).*prior ./...
    sum(sum(local_Pt_given_parsANDx(univ, r, idx_cnt).*prior));


end

%% LOCAL
function p = local_Pt_given_parsANDx(univ_p_resp, r, idxCNT)
% probability of r given the present coefficients
% other important step. Given the response obtained, this function selects,
% from the 4D matrix of all possible events, the 3D matrix corresponding to
% the present SOA.
% Notably, the p values are converted according to the response of the
% observer (p if correct, 1-p if wrong)

if r
    p = squeeze(univ_p_resp(:,:,idxCNT));
else
    p = 1-squeeze(univ_p_resp(:,:,idxCNT));
end

end

