function textureidx = do_GABOR(ptb_win, va, pixXva, cpd, tilt, cnt, bckCnt)
%% textureidx = do_GABOR(radius, pixXva, cpd, tilt, contrast, back_col)
%
% ALL ENTRIES ARE MANDATORY
%
% SYNOPSIS
% ptb_win   -->    window where the texture is created
% va        -->    radius of gabor in visual angles (e.g. 2);
% pixXva    -->    pixels per visual angle (e.g. 56);
% cpd       -->    cycles per degree of visual angle (e.g. 5);
% tilt      -->    gabor orientation (e.g. 45°);
% cnt       -->    contrast (e.g. RGB [0-255])


%% extract number of pixel of figure's edge
% and convert RGB into [0 1] interval
n_pixel = 2*va*pixXva+1; 
unityCnt = cnt/255;
gauss_tail = .0015;
unityBck = bckCnt/255;
%% prepare grid
[x_sin, y_sin] = deal(linspace(-va, va, n_pixel));
[X, Y] = meshgrid(x_sin, y_sin);

%% define sd of gaussian based on the number of pixels
% in order to dynamically change the extension of borders
% rule of thumb: set the borders to encompass 2sd of the gaussian filter 
gauss_p_lim = gauss_tail/unityCnt; % this re-weights the probability values according to the contrast. (.025 at contrast=1)
FH.mys = @(xlim, m, yp) (xlim-m)./(sqrt(2)*erfinv(1-2*yp));
mysd = FH.mys(va, 0, gauss_p_lim); % visual angles, mean, p(X<=x)

% apply arbitrary correction sd<0 (really low contrast values) and give a
% warning
if mysd<0 || isnan(mysd)
    mysd = local_correct_sd(unityCnt, FH, gauss_tail, va);
    too_lsd = true; % too large standard deviation flag
else
    too_lsd = false;
end

%% step 1: draw the dynamic gaussian filter
[gauss_filt_cutoff, x_cut] = local_draw_gauss_filter(x_sin, 0, mysd, gauss_p_lim);

% uncomment following 3 lines to plot the 3d filter
% figure
% surf(X,Y,gauss_filt_cutoff)
% title('gaussian filter with 2sd cutoff')

%% step 2: add a circular logical mask of the radius of 2sd
% to the gabor in order to avoid edges effect in the final image

xylim = x_cut(2);  
if too_lsd; xylim = va; end % consider the correction scenario
radius = xylim^2; 
lgcl_circular_mask = double((X.^2 + Y.^2)<radius);

new_gauss_filt = gauss_filt_cutoff.*lgcl_circular_mask;

% uncomment following 3 lines to plot the 3d filter
% figure
% surf(X,Y,new_gauss_filt)
% title({'gaussian filter with 2sd cutoff', '+ circular filter'})


% design gabor
grating = local_draw_grating(X, Y, cpd, unityCnt, tilt, unityBck);

% uncomment following 3 lines to plot the 3d grating
% figure
% surf(X,Y,grating)
% title({'grating'})

% create uint8 mat
GAB = uint8(255*(unityBck + grating.*new_gauss_filt));

% create texture
textureidx = Screen('MakeTexture', ptb_win, GAB);

end

function [gauss_filt, x_cut] = local_draw_gauss_filter(x_sin, mu, sigma, cutoff)
% design gaussian filter for gabor

% create probability distribution object
pm = makedist('normal', 'mu', mu, 'sigma', sigma); 

% normalize pdf=1 on top of the pdf
y_norm = pdf(pm, x_sin)/max(pdf(pm, x_sin));
   
% get the x values for cutoff on inverse cdf
x_cut = icdf(pm, [cutoff 1-cutoff]);

% create filter
gauss_filt = (y_norm'*y_norm);

end

function grating = local_draw_grating(X, Y, freq, cnt, tilt, unityBck)

% convert tilt in radian
tilt = tilt*pi/180;

% apply formula in Lu&Dosher 2014
grating = unityBck*cnt*sin(2*pi*freq*(Y*sin(tilt)+X*cos(tilt)));

end

function mysd = local_correct_sd(iCnt, FH, gauss_tail, va)
% problem: if contrast RGB is too low, the function for sigma behaves weirdly,
% yielding negative sigma. The whole function in these values has a
% hyperbpolic trend. This is motivated from the current adaptation of the
% formula to obtain sigma.
% solution: compute current sigma based on slope and intercept of the line
% cpassing through the two last points for which the sd still made sense.

xvals_contrast = linspace(0, 1, 256);
xvals_prob = gauss_tail./xvals_contrast;

% find the first 2 values which will not get a negative sigma.
idx_firsts = find(xvals_prob<.5, 2);

% find all sigmas
ysigmas = FH.mys(va, 0, xvals_prob); 


x1 = xvals_contrast(idx_firsts(1)); x2 = xvals_contrast(idx_firsts(2));
y1 = ysigmas(idx_firsts(1)); y2 = ysigmas(idx_firsts(2));

m = (y1 -y2)/(x1 - x2);
q = (x1*y2 - x2*y1)/(x1 - x2);
mysd = m*iCnt+q;

fprintf('\n #### BE CAREFUL #### \n')
fprintf('\n the current contrast value is quite low, applying correction\n')
warning('function might misbehave')
fprintf('\nsee local_correct_sd for a more detailed explanation of the issue\n')

end
