function [P, ploteasy] = cluster_ts_permute(mat, xtime)

warning('the function still requires better implementation')

nperms = 10000;
alpha_t = .05;
alpha_clust = .05;
tail = 1;                   % 0:left, 1:right, 2:both
plotflag = false;
plotdebugflag = false;

nsubjs = size(mat, 2);

% define anonymous function for t calculation
anon_t = @(x) sqrt(nsubjs)*mean(x, 2)./std(x, [], 2);

% compute the current tvalues
exp_t = anon_t(mat);

% compute critical value
crit_t = tinv(1-alpha_t, nsubjs-1);

%% positive clusters
if (tail == 1) || (tail == 2)
    
    % find values exceeding critical value
    lgc_pos = exp_t>crit_t;

    % find cluster
    clust_pos = bwlabel(lgc_pos); 

    % find mass
    mass_clust_pos = local_clustermass(clust_pos, exp_t);

    if (tail == 1)
    
        mass_clust_neg = [];
        
    end
    
end

%% negative clusters
if (tail == 0) || (tail == 2)
    
    % find values exceeding critical value
    lgc_neg = exp_t<-crit_t;
    
    % find cluster
    clust_neg = bwlabel(lgc_neg);  

    % find mass
    mass_clust_neg = local_clustermass(clust_neg, exp_t);
    
end

if (tail == 0)

    mass_clust_pos = [];

end

% permute
rng(1)
rand_dist = nan(nperms, 1);


for iperm = 1:nperms
    
    % 1. select a subsample of participants
    subsmpl = randi([0, 1], nsubjs, 1);
    idxssubsmpl = subsmpl==1;
    % 2. multiply *-1 the subset of participants
    shffld = mat;
    shffld(:, idxssubsmpl) = -shffld(:, idxssubsmpl);

    % compute tvals
    swap_t = anon_t(shffld);
        
    % find clusters
    swap_pos = swap_t>crit_t;
    swap_neg = swap_t<-crit_t;
    
    swap_cl_pos = bwlabel(swap_pos);
    swap_cl_neg = bwlabel(swap_neg);    
    
    lab_pos = unique(swap_cl_pos);
    lab_neg = unique(swap_cl_neg);
    
    % positive cluster
    if any(lab_pos)
        
        pos_mass = nan(max(lab_pos), 1);
        for ipos = 1:max(lab_pos)
            
            swap_lgc_pos = swap_cl_pos==ipos;
            pos_mass(ipos) = sum(swap_t(swap_lgc_pos));
            
        end
        
    else
        
        pos_mass = 0;
        
    end
    
    % negative cluster
    if any(lab_neg)
        
        neg_mass = nan(max(lab_neg), 1);
        for ineg = 1:max(lab_neg)
            
            swap_lgc_neg = swap_cl_neg==ineg;
            neg_mass(ineg) = sum(swap_t(swap_lgc_neg));
            
        end
        
    else
        
        neg_mass = 0;
        
    end
                 
    if tail == 1
        neg_mass = [];
    elseif tail == 0
        pos_mass = [];
    end
    
    % find the maxima between the two
    swap_masses = [pos_mass; neg_mass]; 
    maxabs = max(abs(swap_masses));
    if maxabs==0
        
        rand_dist(iperm) = 0;
    
    else
        
        rand_dist(iperm) = swap_masses(abs(swap_masses)==maxabs);
    
    end
end

%% if debug
% plot figure of random permutation and current statistic

if plotdebugflag
    
    figure; hold on
    histogram(rand_dist)
     
end


% get ecdf for P computation
[cdfranddist_(:,1), cdfranddist_(:,2)] = ecdf(rand_dist);

if ~isempty(mass_clust_pos)
    
    for ipos = 1:size(mass_clust_pos,1)
    
        mass_clust_pos(ipos,2) = 1-cdfranddist_(find(cdfranddist_(:,2)<mass_clust_pos(ipos, 1), 1, 'last'), 1);

    end
    
else
    
    mass_clust_pos(1, 2) = 1;
    
end
    
if ~isempty(mass_clust_neg)

    for ineg = 1:size(mass_clust_neg, 1)
    
        mass_clust_neg(ineg,2) = cdfranddist_(find(cdfranddist_(:,2)<mass_clust_neg(ineg, 1), 1, 'last'), 1);

    end
    
else
    
    mass_clust_neg(1, 2) = 1;
        
end

if plotflag 
    
    if any(mass_clust_pos(:, 2)<alpha_clust)
    
        local_plot_ts(mat, xtime, clust_pos, mass_clust_pos, alpha_clust)
        
    elseif any(mass_clust_neg(:, 2)<alpha_clust)
    
        local_plot_ts(mat, xtime, clust_neg, mass_clust_neg, alpha_clust)

    end
    
end

P = [mass_clust_pos; mass_clust_neg];

%% pipe out things for an easier plot outside the present function

ploteasy.avg = mean(mat, 2);
ploteasy.stderr = std(mat, [], 2)./sqrt(nsubjs);
ploteasy.time = xtime;
ploteasy.tvals = exp_t;

if any(mass_clust_pos(:, 2)<alpha_clust)
    
    idxs_sign = find(mass_clust_pos(:, 2)<alpha_clust);
    clust_pos_cell = cell(numel(idxs_sign), 1);
    
    for iIdx = 1:idxs_sign
        
        swap_p = clust_pos;
        swap_p(swap_p~=iIdx) = nan;
        clust_pos_cell{iIdx} = swap_p;
        
    end
    
    ploteasy.pos_clusts = clust_pos_cell;
    
elseif any(mass_clust_neg(:, 2)<alpha_clust)

    idxs_sign = find(mass_clust_neg(:, 2)<alpha_clust);
    clust_neg_cell = cell(numel(idxs_sign), 1);
    
    for iIdx = 1:idxs_sign
        
        swap_n = clust_neg;
        swap_n(swap_n~=iIdx) = nan;
        clust_neg_cell{iIdx} = swap_n;
        
    end

    ploteasy.neg_clusts = clust_neg_cell;
    
else
    
    ploteasy = [];
    
end



end

%% ########################### LOCAL FUNCTIONS ###########################

function mass_clust = local_clustermass(clust, tvals)

nclusts = max(clust);

mass_clust = nan(nclusts,1 );
for iclust = 1:nclusts
    
    this_clust = clust == iclust; 
    mass_clust(iclust) = sum(tvals(this_clust));
    
end

end

function local_plot_ts(mat, xtime, clust, mass_clust, alpha_clust)

foo = 1;

GA = mean(mat, 2);
Gstderr = std(mat, [], 2)/sqrt(size(mat, 2));

shadedErrorBar(xtime, GA, Gstderr, 'lineProps',  'b'); hold on;
% ylim([-.02 .1])
plot(xtime, zeros(size(xtime)), 'r')

peakgraph = max(GA+Gstderr); line_sign = ones(size(xtime))*(5*peakgraph/4);

idx_sign = find(mass_clust(:,2)<alpha_clust)';

for icl = idx_sign
    
    mask_clust = clust == icl;
    sw = line_sign;
    sw(~mask_clust) = nan;
    
    plot(xtime, sw, 'k', 'LineWidth', 5)
    
end


end
