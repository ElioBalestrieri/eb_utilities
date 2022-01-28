function plot_topographies(cfg, dat, map_sign)

% plot multiple topographies with eeglab topoplot


chanlocs = cfg.chanlocs;

n_subplotrows = ceil(cfg.nsteps/5);

% get the nuber of steps from cfg, and create a consistent number of masks
lsign = length(cfg.timeline); frac_lsign = lsign/cfg.nsteps;

if ~isfield(cfg, 'zlim')
    zlim = [-4, 4];
else
    zlim = cfg.zlim;
end


if isfield(cfg, 'remap')
    
    swap_dat = zeros(length(chanlocs), size(dat, 2));
    
    swap_dat(cfg.remap, :) = dat;
    
    dat=swap_dat;
    
    swap_map = map_sign;
    
    map_sign = false(size(swap_dat));
    
    map_sign(cfg.remap, :) = swap_map;
    
end

% dat(~map_sign) = 0;

%%
endpoint = 0; cmap = colormap('parula');
for istep = 1:cfg.nsteps
    
    strtpoint = endpoint + 1;
    endpoint = strtpoint + floor(frac_lsign);
    
    if endpoint > lsign
        
        endpoint = lsign;
        
    end
    
    current_mat = dat(:, strtpoint:endpoint);
    current_clustmat = map_sign(:, strtpoint:endpoint);
    
    chans_clusterpart = any(current_clustmat, 2);
    chanidxs = find(chans_clusterpart);
    
    to_show = mean(current_mat, 2);
    
    subplot(n_subplotrows, 5, istep)
    topoplot(to_show, chanlocs, ...
        'maplimits', zlim,...
        'conv', 'on',...
        'emarker2', {chanidxs, 'o', 'w', 5}, ...
        'colormap', cmap, ...
        'whitebk', 'on'...
        )
    
    title(sprintf('t = [%4.3f ; %4.3f]', cfg.timeline(strtpoint), cfg.timeline(endpoint)))

    
end
