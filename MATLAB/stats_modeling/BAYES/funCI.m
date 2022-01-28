function CI = funCI(parvals_pdf, vals_x,  realval, iperm)

% determine empirical cdf
mat_mult = tril(ones(numel(parvals_pdf)));
cdf_y = mat_mult*parvals_pdf;

% interpolate points to achieve higher accuracy
new_xvals = vals_x(1):.0001:vals_x(end);
new_ycdf = interp1(vals_x, cdf_y, new_xvals);


% determine confidence intervals
CI = nan(1,6);

lower = new_xvals(find(new_ycdf>=.025, 1));
if ~isempty(lower)
    CI(1) = lower;
    CI(5) = nan;
else
    CI(1) = min(vals_x);
    warning('lower CI bound exceeds range of possible values')
    CI(5) = iperm;
    fprintf('out of range for perm %i', iperm)
end

upper = new_xvals(find(new_ycdf<=.975, 1, 'last'));
if ~isempty(upper)
    CI(2) = upper;
    CI(6) = nan;
else
    warning('upper CI bound exceeds range of possible values')
    CI(2) = max(vals_x);
    CI(6) = iperm;
    fprintf('out of range for perm %i', iperm)
end

CI(3) = double((realval >=CI(1)) & (realval <= CI(2)));
CI(4) = realval;

end

