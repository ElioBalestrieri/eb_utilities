function wv = optimized_wavelet(data, srate, CFG)
% Wavelet transform function based on the work by MX Cohen
% function implemented by Niko Busch
% optimized by Elio Balestrieri
%
% data -> EEG data in 3D: chansXtimeXtrials
% srate -> sampling rate
% CFG -> structure with MANDATORY subfields:
%        .min_freq 
%        .max_freq
%        .num_frex
%
%     -> OPTIONAL subfields:
%        .frex: the vector of the frequencies 
%        .fwhm_t (minimal advised 1/frex, default 2/frex)
%        .nextpow2: default false. speeds up computation, but ringing
%        artifacts?
%     -> .ITC: compute ITC; default: false
%     -> .keeptrials: keep the single trials, default false

% 18-Jun-2021
% set default for "CFG.nextpow2 = false" due to ringing artifact (eb)
% 16-Aug-2021
% set CFG fields for ITC and keeptrials (for flexibility/performance tradeoff) 



% Make sure the data are three dimensional: chans * time * trials.
if ndims(data) == 1
    data(1,1,:) = data;
elseif ndims(data) == 2 %#ok<ISMAT>
    data(1,:,:) = data;
end

% set default for no zero pad
if ~isfield(CFG, 'nextpow2')
    CFG.nextpow2 = false;
end

% complete missing inputs
% compared to orgiginal call to Niko's function)
if ~isfield(CFG, 'frex')
    CFG.frex = linspace(CFG.min_freq, CFG.max_freq, CFG.num_frex);
end

if ~isfield(CFG, 'fwhm_t')
    
    CFG.fwhm_t = 2./CFG.frex; 

end

if ~isfield(CFG, 'ITC')
    CFG.ITC = false;
end

if ~isfield(CFG, 'keeptrials')
    CFG.keeptrials = false;
end


if isfield(CFG, 'resample')
    if CFG.resample
        new_srate = ceil(2.5*CFG.max_freq); %(1/2 over nyquist)
        new_data = resample(double(data), new_srate, srate, 'Dimension', 2);
        data = new_data;
        srate = new_srate;
    end
end



[nchans, npnts, ntrials] = size(data);


% internal computation of CFG parameters dependent on fundamental inputs
nfreqs = CFG.num_frex;

% Prepare wavelets.
wv = local_get_wavelets(CFG, srate, npnts, ntrials);

% concatenate data along the time dimension
long_data = reshape(data, nchans, npnts*ntrials);

% compute the FFT
if CFG.nextpow2
    FFT_longdata = fft(long_data, wv.nConvPow2, 2);
else
    FFT_longdata = fft(long_data, wv.nConv, 2);
end 

% preallocate zeros for pow and itc
if CFG.keeptrials
    
    wv.pow = single(zeros(nfreqs, npnts, nchans, ntrials));
    wv.itc = nan;
    
    if CFG.ITC
        
        wv.itc = complex(single(zeros(nfreqs, npnts, nchans, ntrials)));
        
    end

else
    
    wv.pow = single(zeros(nfreqs, npnts, nchans));
    wv.itc = nan;
    
    if CFG.ITC
        
        wv.itc = single(zeros(nfreqs, npnts, nchans));

    end

end
    
    
% loop over frequencies
for ifreq = 1:nfreqs
    
    % expand wavelet to allow matrix multiplication
    this_wav = wv.waveletX{ifreq};
    wavmat = repmat(this_wav, nchans, 1);
    
    % actually compute multiplication in freq domain
    as = ifft(FFT_longdata .* wavmat, [], 2);
    as = as(:, 1:wv.nConv);
    as = as(:, wv.half_wave+1:end-wv.half_wave);
    
    % bring data back in original form, then change dimord for
    % compatibility with 4D representation
    as = reshape(as, nchans, npnts, ntrials);
    as = permute(as, [2, 1, 3]);

    % store power (and ITC) in average or single trials, as requested)
    if CFG.ITC
        
        temp_abs = abs(as);
        nrmzd_as = as ./ temp_abs;
        temp_pow = temp_abs.^2;
        
    else
        
        temp_pow = abs(as) .^ 2;
        
    end
       
    if CFG.keeptrials
        
        wv.pow(ifreq,:,:,:) = temp_pow;

        if CFG.ITC
            
            wv.itc(ifreq,:,:,:) = nrmzd_as;

        end
        
    else
        
        wv.pow(ifreq,:,:) = mean(temp_pow, 3);

        if CFG.ITC
            
            wv.itc(ifreq,:,:) = abs(mean(nrmzd_as, 3));

        end
        
    end
         
end


end

%% ######################## LOCAL FUNCTIONS

function wv = local_get_wavelets(CFG, srate, npnts, ntrials)

% The time axis of the wavelet.
wv.wavtime = -2:1/srate:2;
wv.half_wave = (length(wv.wavtime)-1)/2;

% general parameters to perform the tf
wv.nWave = length(wv.wavtime);
nData = npnts * ntrials;
wv.nConv = wv.nWave + nData - 1;
wv.nConvPow2 = pow2(nextpow2(wv.nConv));

% preallocate cell for wavelets
wv.waveletX = cell(CFG.num_frex, 1);

for ifreq = 1:CFG.num_frex       
    
    % create Gaussian window
    this_gaus_win = exp((-4 * log(2) * wv.wavtime .^2) / (CFG.fwhm_t(ifreq)^2) ); 
     
    % create wavelet and get its FFT
    % the wavelet doesn't change on each trial...
    this_wavelet = exp(2 * 1i * pi * CFG.frex(ifreq) .* wv.wavtime) .* ...
        this_gaus_win;
    
    if CFG.nextpow2
        freq_domain_wave = fft(this_wavelet, wv.nConvPow2);
    else
        freq_domain_wave = fft(this_wavelet, wv.nConv);
    end
    
    wv.waveletX{ifreq} = single(freq_domain_wave ./ max(freq_domain_wave));
    
end

end