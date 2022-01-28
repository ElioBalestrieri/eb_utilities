function CSP_struct = CSP_computation(bp_signal, lgcl, LmR_cnts)
%% apply the procedure as described by blankertz et al. 2008
    
% train labels
lgcl_cond1 = lgcl.train.cond1;
lgcl_cond2 = lgcl.train.cond2;

% transfer labels
lgcl_trns1 = lgcl.transfer.cond1;
lgcl_trns2 = lgcl.transfer.cond2;

% %step 1: center and scale data, and compute covariance matrices
ntrls = size(bp_signal, 3);
nchans = size(bp_signal, 1);
ntp = size(bp_signal, 2); 

scaled_dat = nan(size(bp_signal));
cov_dat = nan(nchans, nchans, ntrls);

for itrl = 1:ntrls
    
    this_trl = bp_signal(:, :, itrl);
    scaled_trl = bsxfun(@minus, this_trl, mean(this_trl, 2)); 
    scaled_dat(:, :, itrl) = scaled_trl;
    temp = scaled_trl*scaled_trl'./(ntp-1);  % calculate covariance as X*X' for each trial (divided by the number of timepoints -1)
    cov_dat(:, :, itrl) =  temp;
    
end

%% step 2: select cov mat subsets for the 2 different conditions
cov_1 = mean(cov_dat(:, :, lgcl_cond1), 3);
cov_2 = mean(cov_dat(:, :, lgcl_cond2), 3);

%% step 3: solve generalized eigenvectors problem
[W, D] = eig(cov_1, cov_1 + cov_2);

%% step 4: choose J eigenvectors from both ends and extract the features for SVM
J = 3;
features = local_get_features(D, W, cov_dat, nchans, ntrls, J); % computes features of the whole set for generalization purpose
features_loo = local_get_features_loo(lgcl_cond1, lgcl_cond2,...
    cov_dat, nchans, J); % compute just the features for the training set, with a loo approach, in order to get a correct estimate of accuracy

% some cases show imaginary value close to machine precision. Neglect this
% cases by taking only the real part.
if ~isreal(features)
    
    features = real(features);
    W = real(W);
    D = real(D);
    
end

if ~isreal(features_loo)
    
    features_loo = real(features_loo);
    
end

%% step 5: train classifier
% training on the loo, to avoid overfitting.
trainset_lgcl = lgcl_cond1 | lgcl_cond2;
red_lbls = lgcl_cond1(trainset_lgcl);
testset_lgcl = lgcl_trns1 | lgcl_trns2;

% optimizing the SVM on subj 1 showed that the best Hyperparameters were:
% kernelFunction = linear
% BoxConstraint = 36.725
% KernelScale = 32.124

% fit model
rng(0) % for reproducibility
svm_mdl_loo = fitcsvm(features_loo, red_lbls, 'KernelFunction', 'linear', ...
    'BoxConstraint', 36.725, 'KernelScale', 32.124); 
% crossvalidate model
cv_svm_mdl = crossval(svm_mdl_loo);
accuracy_mdl = 1-kfoldLoss(cv_svm_mdl);
assigned_lab_crossval = kfoldPredict(cv_svm_mdl);

%% step 7: construct output structure 
% prob1: the loo procedure yields naturally N patterns, and N spatial
% filter mtrices, where N is the number of trial to be taken into account. 
% quest1: how could we show a topography of the patterns of interest?
% answer: I'll pipe out the W and A coming out for the CSP considering all
% trials.
CSP_struct.Jperside = J;
% put here loo accuracy
CSP_struct.acc_mdl_loo = accuracy_mdl;

% prob2: the accuracy yielded by the previous SVM is referring to the loo.
% Let's recompute the accuracy of the non loo
red_features_nocv = features(trainset_lgcl, :);
svm_mdl_nocv = fitcsvm(red_features_nocv, red_lbls, 'KernelFunction', 'linear', ...
    'BoxConstraint', 36.725, 'KernelScale', 32.124); 
% crossvalidate model to get SVM accuracy
nocv_svm_mdl = crossval(svm_mdl_nocv);
accuracy_mdl_nocv = 1-kfoldLoss(nocv_svm_mdl);

%% step 6: apply the model on all trials to obtain labels
% this is the model trained on all training trials (hence, no loo). 
assigned_labels = predict(svm_mdl_nocv, features);
% to account for overfitting, substitute in the training entries the labels
% obtained by crossval on the leave one out approach
assigned_labels(trainset_lgcl) = assigned_lab_crossval;

CSP_struct.acc_mdl_nocv = accuracy_mdl_nocv;

% at this point, compute even the transfer accuracy of the last model
% (now the loo is not a problem anymore, since the generalization includes
% by definition trials that were not included in the covariance matrix
% definition)
test_lab = assigned_labels(testset_lgcl);
CSP_struct.acc_transfer = mean(lgcl_trns1(testset_lgcl) == test_lab);

if isfield(lgcl, 'cntflag')
    
    if lgcl.cntflag
        
        % evaluate accuracy at different contrast bins (normalized)
        red_cnts = LmR_cnts(trainset_lgcl);
        acc_preds = assigned_lab_crossval == red_lbls;
        CSP_struct.acc_cnt_binned = local_acc_binned(acc_preds , red_cnts);
        CSP_struct.cnt_accurate_preds = acc_preds;
        
        % create a matrix of scrambled features AND contrast differences
        swap_feats = features_loo;
        ntrls = size(swap_feats, 1);

        for icol = 1:J*2

            swap_feats(:, icol) = randsample(swap_feats(:, icol), ntrls);

        end

        swap_cnts = randsample(LmR_cnts, ntrls);

        % define feature models
        dummy = [swap_cnts, swap_feats];
        cnt_only = [red_cnts, swap_feats];
        CSP_only = [swap_cnts, features_loo]; %  
        full_mdl = [red_cnts, features_loo];

        %% dummy
        svm_dummy_mdl = fitcsvm(dummy, red_lbls, 'KernelFunction', 'linear', ...
            'BoxConstraint', 36.725, 'KernelScale', 32.124); 
        % crossvalidate model
        cv_svm_dummy_mdl = crossval(svm_dummy_mdl);
        accuracy_dummy_mdl = 1-kfoldLoss(cv_svm_dummy_mdl);

        %% contrast
        svm_cnt_mdl_loo = fitcsvm(cnt_only, red_lbls, 'KernelFunction', 'linear', ...
            'BoxConstraint', 36.725, 'KernelScale', 32.124); 
        % crossvalidate model
        cv_svm_cnt_mdl = crossval(svm_cnt_mdl_loo);
        accuracy_cnt_mdl = 1-kfoldLoss(cv_svm_cnt_mdl);

        %% CSP
        svm_CSP_msl = fitcsvm(CSP_only, red_lbls, 'KernelFunction', 'linear', ...
            'BoxConstraint', 36.725, 'KernelScale', 32.124); 
        % crossvalidate model
        cv_svm_mdl_CSP = crossval(svm_CSP_msl);
        accuracy_mdl_CSP = 1-kfoldLoss(cv_svm_mdl_CSP);

        %% full model
        svm_full_mdl = fitcsvm(full_mdl, red_lbls, 'KernelFunction', 'linear', ...
            'BoxConstraint', 36.725, 'KernelScale', 32.124); 
        % crossvalidate model
        cv_svm_FULL_mdl = crossval(svm_full_mdl);
        accuracy_FULL_mdl = 1-kfoldLoss(cv_svm_FULL_mdl);

        % put here loo accuracy
        CSP_struct.accuracy_dummy_mdl = accuracy_dummy_mdl;
        CSP_struct.accuracy_cnt_mdl = accuracy_cnt_mdl;
        CSP_struct.accuracy_CSP_mdl = accuracy_mdl_CSP;
        CSP_struct.accuracy_full_mdl = accuracy_FULL_mdl;
        
    end
        
else
    
    % evaluate accuracy at different contrast bins (normalized)
    red_cnts = LmR_cnts(testset_lgcl);
    acc_preds = lgcl_trns1(testset_lgcl) == test_lab;
    CSP_struct.acc_cnt_binned = local_acc_binned(acc_preds , red_cnts);
    CSP_struct.cnt_accurate_preds = acc_preds;
   
end
        

end

%% ######################### LOCAL FUNCTIONS ##########################

function features = local_get_features(D, W, cov_dat, nchans, ntrls, J)

W_resh = [diag(D)'; W]; W_resh = sortrows(W_resh', 1)'; W_resh(1, :) = [];

% select eigenvectors of interest
Jidxs = [1:J, (nchans-J+1):nchans];
nj = J*2;
sel_W = W_resh(:,Jidxs);

features = nan(ntrls, nj);
for itrl = 1:ntrls
    
    this_trl = cov_dat(:, :, itrl);
    features(itrl, :) = log(diag(sel_W'*this_trl*sel_W));
       
end

end

function features_loo = local_get_features_loo(lgcl_cond1, lgcl_cond2,...
    cov_dat, nchans, J)

trainset_lgcl = lgcl_cond1 | lgcl_cond2;

ntrain = sum(trainset_lgcl);

features_loo = nan(ntrain, 2*J);

% now, compute the features with a leave one out approach, in order to
% avoid overfitting coming from taking the current trial into the mean of
% covariance matrices

for itr = 1:ntrain
    
    red_cov = cov_dat(:, :, trainset_lgcl);  % select only the trainset
    
    curr_trl = red_cov(:, :, itr); % select the current trial of interest for features compution
    red_cov(:, :, itr) = [];       % leave that trial out in the covariance matrices
    
    red_lgcl.cond1 = lgcl_cond1(trainset_lgcl); red_lgcl.cond1(itr) = []; % null logical labels for the present trial
    red_lgcl.cond2 = lgcl_cond2(trainset_lgcl); red_lgcl.cond2(itr) = [];
    

    cov_1 = mean(red_cov(:, :, red_lgcl.cond1), 3);
    cov_2 = mean(red_cov(:, :, red_lgcl.cond2), 3);

    % step 3: solve generalized eigenvectors problem
    [W, D] = eig(cov_1, cov_1 + cov_2);

    % step 4: obtain features for the current trial
    features = local_get_features(D, W, curr_trl, nchans, 1, J);

    features_loo(itr, :) = features;
    
     
     
end

 
end

function acc_cnt_binned = local_acc_binned(acc_preds, red_cnts)

nranks = 3;

rnks_lbls = quantileranks(zscore(red_cnts), nranks);
acc_cnt_binned = nan(nranks, 1);

for irank = 1:nranks
    
    maskrank = rnks_lbls == irank;
    acc_cnt_binned(irank) = mean(acc_preds(maskrank));
    
end

end

