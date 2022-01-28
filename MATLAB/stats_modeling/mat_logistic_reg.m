function [B, out_struct] = mat_logistic_reg(DATA, labels, cfg)
%% B = mat_logistic_reg(data_mat, labels, cfg)
% the function performs logistic regression on data matrices, given data
% and labels. Parameters estimation is implemented through IRLS (Bishop, 2006) 
%
% DATA      <- [i x j x predictors x repetitions] matrix. It has to contain
%              the intercept term of ones (and you have to remember where
%              it was in the matrix)
% labels    <- [i x j x repetitions] matrix. Can be output of repmat.
% cfg       <- configuration structure containing:
%    .tolerance <- when the weight change should stop (default 1e-7)
%    .iter  <- n algorithm's iterations (default 50)
%
% created by Elio Balestrieri on 20-Feb-2020

%% obtain dim size
datsize = size(DATA);
nregs = prod(datsize([1 2]));
npreds = datsize(3);
nrep = datsize(4);

% normalize for normalized beta scores
if cfg.normalize
    
    DATA(:, :, 2:end, :) = zscore(DATA(:, :, 2:end, :),[], 4); % don't normalize the intercept term!
    
end

X_tot = nan(nrep, npreds, nregs);
labtot = nan(nrep, nregs);

for irep = 1:nrep
    
    sw = labels(:, :, irep);
    labtot(irep, :) = sw(:);
    
    for ipred = 1:npreds
        
        red = squeeze(DATA(:, :, ipred, irep)); 
        X_tot(irep, ipred, :) = red(:);
        
    end
              
end

%% define the function
sigmfun = @(x) 1./(1+exp(-x));

%% start fitting (IRLS)
if ~isfield(cfg, 'niter')
    it_all = 50;
else
    it_all = cfg.niter;
end

if ~isfield(cfg, 'tolerance')
    tolerance = 1e-7;
else
    tolerance = cfg.tolerance;
end

% preallocate B matrix (2d)
B = nan(datsize(1), datsize(2), npreds);
% preallocate t-y mat
sqrdErr = nan(datsize(1), datsize(2));
% preallocate t-y cell
distTarget = cell(nregs,1);

% store the number of iterations reached for the whole data
vect_iters = nan(nregs, 1);


% loop for all regressions that have to be performed
for ireg = 1:nregs

    w = zeros(npreds, 1);
    
    t = labtot(:, ireg);
    X = squeeze(X_tot(:, :, ireg));
    
    % actual IRLS algorithm loop
    it = 0;
    while it <= it_all

        it = it +1;
        
        y = sigmfun(X*w);
        R = diag(y.*(1-y));
        z = X*w - R\(y-t);        
        w_new = (X'*R*X)\X'*R*z;
        
        change = sqrt(sum((w-w_new).^2));

        [warnmsg, msgid] = lastwarn;
        if strcmp(msgid,'MATLAB:singularMatrix') || strcmp(msgid,'MATLAB:illConditionedMatrix')
            
            foo = 1; % for debugging purposes
            
            
        end
        
        
        w = w_new;

        if change < tolerance
            
            break
            
        end
        
    end

    vect_iters(ireg) = it;
    max_iters = vect_iters == it_all;
    if any(max_iters)
        
        fprintf('\n max iter reached %i times for the current subject\n', sum(max_iters))
        
    end
    
    [rn, cn] = ind2sub(datsize(1:2), ireg);
    B(rn, cn, :) = w;
    sqrdErr(rn, cn) = sum((y-t).^2); 
    distTarget{ireg} = y;
    
end


%% sandbox

if size(B, 3) == 4

    B_alpha = B(:, :, 3);
    B_inter = B(:, :, 4);

    if isfield(cfg, 'timelogic')

        B_alpha = B_alpha(:, cfg.timelogic);
        B_inter = B_inter(:, cfg.timelogic);    

    end


    [lin_idx_maxpow, sout_maxpow] = local_select(B_alpha, 'pos');
    [lin_idx_minpow, sout_minpow] = local_select(B_alpha, 'neg');
    [lin_idx_mininter, sout_minint] = local_select(B_inter, 'neg');

    cnts = X(:, 2);
    alpha_max = X_tot(:, 3, lin_idx_maxpow);
    alpha_min = X_tot(:, 3, lin_idx_minpow);
    int_min = X_tot(:, 3, lin_idx_mininter);


    labassigned_tMAX = distTarget{lin_idx_maxpow};
    labassigned_tMIN = distTarget{lin_idx_minpow};
    labassigned_interMIN = distTarget{lin_idx_mininter};


    nbins_alpha = 2;
    nbins_cnt = 5;


    cnts_quant = quantileranks(cnts, nbins_cnt);
    alpha_quant_MAX = quantileranks(alpha_max, nbins_alpha);
    alpha_quant_MIN = quantileranks(alpha_min, nbins_alpha);
    int_quant_MIN = quantileranks(int_min, nbins_alpha);


    [cont_table_max, cont_table_min, cont_table_min_inter] = deal(nan(nbins_alpha, nbins_cnt));

    for iAlpha = 1:nbins_alpha

        mask_alpha_max = alpha_quant_MAX == iAlpha;
        mask_alpha_min = alpha_quant_MIN == iAlpha;
        mask_int_min = int_quant_MIN == iAlpha;


        for iCnt = 1:nbins_cnt

            mask_cnt = cnts_quant == iCnt;


            ass_labels_max = labassigned_tMAX(mask_alpha_max & mask_cnt);
            ass_labels_min = labassigned_tMIN(mask_alpha_min & mask_cnt);
            ass_labels_min_inter = labassigned_interMIN(mask_int_min & mask_cnt);

            cont_table_min_inter(iAlpha, iCnt) = mean(ass_labels_min_inter);
            cont_table_max(iAlpha, iCnt) = mean(ass_labels_max);
            cont_table_min(iAlpha, iCnt) = mean(ass_labels_min);

        end

    end


    diff_cont_tables = cont_table_max - cont_table_min;

    out_struct = [];
    out_struct.cont_table_max = cont_table_max;
    out_struct.cont_table_min = cont_table_min;
    out_struct.cont_table_min_inter = cont_table_min_inter;
    out_struct.diff_cont_tables = diff_cont_tables;
    out_struct.maxpow = sout_maxpow;
    out_struct.minpow = sout_minpow;
    out_struct.minint = sout_minint;

else
    
    out_struct = [];
    
end
    
% figure; 
% subplot(2, 2, 1)
% plot(cont_table_max')
% subplot(2, 2, 2)
% plot(cont_table_min')
% subplot(2, 2, 3)
% plot(diff_cont_tables')
% subplot(2, 2, 4)
% plot(cont_table_min_inter')
% 
% 
% figure;
% subplot(3, 1, 1)
% plot(sout_maxpow.tl_chan)
% subplot(3, 1, 2)
% plot(sout_minpow.tl_chan)
% subplot(3, 1, 3)
% plot(sout_minint.tl_chan)


end


%% ################# LOCAL FUNCTIONS

function [lin_idx, struct_out] = local_select(mat, string_dir)

overtime_sum = sum(mat, 2);

mat_logic = false(size(mat));

switch string_dir
    
    case 'pos'
        
        chan_idx = find(overtime_sum == max(overtime_sum));
        timeline_chan = mat(chan_idx, :);
        time_idx = find(timeline_chan == max(timeline_chan));
        mat_logic(chan_idx, time_idx) = true;
        
    case 'neg'
        
        chan_idx = find(overtime_sum == min(overtime_sum));
        timeline_chan = mat(chan_idx, :);
        time_idx = find(timeline_chan == min(timeline_chan));
        mat_logic(chan_idx, time_idx) = true;
      
end

lin_idx = mat_logic(:);

struct_out.tl_chan = timeline_chan;
struct_out.chan_idx = chan_idx;
struct_out.time_idx = time_idx;

end