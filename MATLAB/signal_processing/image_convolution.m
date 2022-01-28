function convolved_sig = image_convolution(mat_im, cfg)

% the function takes a matrix of (square) images sized N pixels
% and convolves all the images with a set of gabor kernels of varying
% orientations and spatial frequencies.
% credit to Martin Rolfs and Richard Schweitzer for having taught me a way
% faster way to do this...
%
% cfg
%    .freqspace = frequencies of the gabors (cycles per degree)
%    .anglespace = orientation angles (rad)

% eb, last edit 28 January 2022

ntrls = size(mat_im, 3);
sizeimage = size(mat_im, 1); % assume square images!!!

%% reshape big matrix 
% to allow faster matrix multiplication
mat_im = (permute(mat_im, [3, 2, 1]) - 128) ./ 128;
reshpd = reshape(mat_im, [sizeimage*sizeimage, ntrls]);

%% convolve with kernels
nfreqs = length(cfg.freqspace);
nangles = length(cfg.anglespace);

convolved_sig = nan(nfreqs, nangles, ntrls);

freq_acc = 0;
for thisfreq = cfg.freqspace
    
    freq_acc = freq_acc + 1;
    
    angle_acc = 0;
    for thisangle = cfg.anglespace
        
        angle_acc = angle_acc +1;
        
        par = [];
        par.siz = sizeimage-1;
        par.frq = thisfreq/55;
        par.ori = thisangle;
        par.amp = 1;

        filt1 = local_getGaborPatch(par, 0);
        filt2 = local_getGaborPatch(par, pi/2);
        
        filt_resp = ((filt1(:)'*reshpd).^2 + (filt2(:)'*reshpd).^2) .^ .5;
        
        convolved_sig(freq_acc, angle_acc, :) = filt_resp;
                          
    end
    
    fprintf('\nConvolving signals, done %i/%i', freq_acc, nfreqs)
    
end
        


end

function G = local_getGaborPatch(p, pha, use_gaussian)
%
% 2008 by Martin Rolfs, edited 2019 by Richard Schweitzer

if nargin < 3
    use_gaussian = 0;
end

[X,Y] = meshgrid(-p.siz/2:p.siz/2,-p.siz/2:p.siz/2);
Grating  = p.amp*cos(X*(2*pi*p.frq*cos(p.ori)) + Y*(2*pi*p.frq*sin(p.ori))+ pha) ;
% Circle   = X.^2 + Y.^2 <= (p.siz/2)^2;      % circular boundary
G = Grating; % G = Circle.*Grating


% Gaussian envelope?
if use_gaussian
    Gaussian = exp(-(X.^2+Y.^2)./(2*p.sig^2)); 
    G = G.*Gaussian;
end
end