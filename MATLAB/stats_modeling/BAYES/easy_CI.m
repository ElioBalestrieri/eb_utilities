function CI = easy_CI(this, cfg)

parsnames = {'alpha', 'beta'};

CI = nan(2,numel(parsnames));

for ipar = 1:2
    
    parvals_pdf = this.(parsnames{ipar});
    vals_x = cfg.vals.(parsnames{ipar});

    % determine empirical cdf
    mat_mult = tril(ones(numel(parvals_pdf)));
    cdf_y = mat_mult*parvals_pdf;

    % interpolate points to achieve higher accuracy
    new_xvals = vals_x(1):.0001:vals_x(end);
    new_ycdf = interp1(vals_x, cdf_y, new_xvals);


    lower = new_xvals(find(new_ycdf>=.025, 1));
    CI(1, ipar) = lower;

    upper = new_xvals(find(new_ycdf<=.975, 1, 'last'));
    CI(2, ipar) = upper;
    
end

end

