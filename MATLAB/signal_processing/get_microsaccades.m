function [s_data] = get_microsaccades(s_data, cfg)
% implement MS detection algorithm described in Engbert & Kliegl (2003)
% currently the only difference in th ealgorithm per se is that, since we
% are collecting data from just one eye, we'll skip the selection of
% microsaccadic event happening for both eyes.
% started by EB on 15-Apr-2019

% modified on 25-08-2019 to allow double channel recording

% subfields necessary
% cfg.kernellength = 4;
% cfg.binocular = true;
% cfg.noisesuppress = true;
% cfg.lambda = 5;

% after engbert and ... 2006 PNAS
lambda = cfg.lambda;

xchan_idx = find(ismember(s_data.label, {'both_x'}));
ychan_idx = find(ismember(s_data.label, {'both_y'}));

% xchan_idx = find(ismember(s_data.label, {'left_x', 'right_x'}));
% ychan_idx = find(ismember(s_data.label, {'left_y', 'right_y'}));


% step1 compute velocity
s_data = local_compute_velocity(s_data, xchan_idx, ychan_idx);

% step2 compute threshold
s_data = local_compute_threshold(s_data, lambda);

% step 3 select MS events
s_data = local_select_MS(s_data, cfg);

% OPTIONAL step 5 consider only time of interest
if isfield(cfg, 'toi')
    
    s_data = local_select_toi(s_data, cfg);
    
end

% step 6 -added by me-
% a MS is an event extended in time. But having logical TRUE for each
% timepoint showing high velocity might be a confounder, especially when we
% consider the angles: for this reason, we add another subfield containing
% only the first sample of the saccadic event

swap_single_saccade_ON = diff(s_data.lgcl_mask_MS)==1;
s_data.lgcl_MS_onset = logical(cat(1, s_data.lgcl_mask_MS(1,:,:),...
    swap_single_saccade_ON));

% it has to be noted that the single MS onset might be werid and not than
% informative. Solution: take even the MS offset, and compute the angle
% between the onset and offset of MS, on the data and not on the speed.
swap_single_saccade_OFF = diff(s_data.lgcl_mask_MS) == -1;
s_data.lgcl_MS_offset = logical(cat(1, swap_single_saccade_OFF, ...
    s_data.lgcl_mask_MS(end,:,:))); % instead of assigning 0s, let's assign the
                                  % last row of the MS, in order to correct
                                  % for MSs started but not conluded on
                                  % time

% ... and only now we compute angles
s_data = local_compute_angles(s_data, xchan_idx, ychan_idx);

% now perform binocular selection
if isfield(cfg, 'binocular')
    if cfg.binocular
        s_data = local_select_binocularMS(s_data);
    end
end

end

%% ######################### LOCAL FUNCTIONS ##############################

function s_data = local_compute_velocity(s_data, xchan_idx, ychan_idx)

totchan = [xchan_idx, ychan_idx]; nchan = numel(totchan);

% create matrices of values for velocity computation
mat_nminus1 = cat(1, nan(1,nchan,size(s_data.trial, 3)), ...
    s_data.trial(1:end-1, [xchan_idx, ychan_idx], :));
mat_nminus2 = cat(1, nan(2,nchan,size(s_data.trial, 3)), ...
    s_data.trial(1:end-2, [xchan_idx, ychan_idx], :));
mat_nplus1 = cat(1, s_data.trial(2:end, [xchan_idx, ychan_idx], :),...
    nan(1,nchan,size(s_data.trial, 3)));
mat_nplus2 = cat(1, s_data.trial(3:end, [xchan_idx, ychan_idx], :),...
    nan(2,nchan,size(s_data.trial, 3)));

deltaT = mean(diff(s_data.x_time));
s_data.velocity = (mat_nplus2 + mat_nplus1 - mat_nminus1 - mat_nminus2)/...
    (6*deltaT);

end

function s_data = local_compute_threshold(s_data, lambda)

% substep 1 --> compute std by applying a median estimator to the TS
% big question: mismatch between 2003 (no square root) and 2006 (sqrt)
% articles. makes more sense to me to have root squared vals though
stdxy = sqrt(nanmedian(s_data.velocity.^2, 1) - ...
    nanmedian(s_data.velocity,1).^2);

% substep 2 --> define threshold
s_data.MSthresh = repmat(stdxy*lambda, size(s_data.velocity,1), 1, 1);

end

function s_data = local_select_MS(s_data, cfg)

% apply formula as in engbert 2006
numX = squeeze(s_data.velocity(:,1,:));
numY = squeeze(s_data.velocity(:,2,:));
denX = squeeze(s_data.MSthresh(:,1,:));
denY = squeeze(s_data.MSthresh(:,2,:));

% now compare info from 2 eyes (if present)
if size(s_data.MSthresh, 2)>2
    
    % still compute the previous mask
    mask_eye_1 = ((numX./denX).^2 + (numY./denY).^2) > 1;
    
    numX2 = squeeze(s_data.velocity(:,3,:));
    numY2 = squeeze(s_data.velocity(:,4,:));
    denX2 = squeeze(s_data.MSthresh(:,3,:));
    denY2 = squeeze(s_data.MSthresh(:,4,:));

    mask_eye_2 = ((numX2./denX2).^2 + (numY2./denY2).^2) > 1;  
    
    s_data.lgcl_mask_MS = cat(4,mask_eye_1, mask_eye_2);

else
    
    s_data.lgcl_mask_MS = ((numX./denX).^2 + (numY./denY).^2) > 1;

end
% noise reduction: 
% the easiest way to implement a selection criterion is by convolving the
% logical with a kernel of ones. This allows you to have a fast way of
% defining MS vs noise
% open question: which kernel length to use? 3 datapoints -hence variable
% in time- or a fixed tw? 

lgcl_kernel = ones(1,cfg.kernellength);

% preallocate for speed
swap_lgcl_conv = nan(size(s_data.lgcl_mask_MS,1),...
    size(s_data.lgcl_mask_MS,2),...
    size(s_data.lgcl_mask_MS,4));

% start for loop in each trial. Ideally, even this loop could be avoided,
% maybe, by using 2dconv. Nevertheless I did not understand how it works,
% and I prefer to stick to the original

% update: keep separated matrices for the two eyes
for iEye = 1:size(s_data.lgcl_mask_MS,4)
    for iTrl = 1:size(s_data.lgcl_mask_MS,2)

        vectConv = conv(squeeze(s_data.lgcl_mask_MS(:,iTrl, iEye)), lgcl_kernel)...
            >=cfg.kernellength;

        % we have to account for the fact that any value reaching the threshold
        % shows the "spike" at a delay due to the convolution process. There is
        % the need to put ones even at the beginning of the MS itself, done by
        % find
        spikesP = find(vectConv); deltaTcorrector = -(0:cfg.kernellength-1);
        fooidx = repmat(spikesP, 1, numel(deltaTcorrector));
        footmd = repmat(deltaTcorrector, numel(spikesP), 1);
        rightIdx = unique(fooidx+footmd);
        vectConv(rightIdx) = true;
        % cut the last N-1 elements, were N is the kernel length
        vectConv((end-cfg.kernellength+2):end) = [];
        swap_lgcl_conv(:,iTrl,iEye) = vectConv;

    end
end
% attach the matrix obtained in this way to the data
% s_data.old_lgcl_mask = s_data.lgcl_mask_MS;
s_data.lgcl_mask_MS = logical(swap_lgcl_conv);

end

function s_data = local_select_toi(s_data, cfg)

lgcl_mask_time = s_data.x_time>=min(cfg.toi) & s_data.x_time<=max(cfg.toi);

% now apply this to the subfields of interest
s_data.trial = s_data.trial(lgcl_mask_time,:,:);
s_data.velocity = s_data.velocity(lgcl_mask_time,:,:);
s_data.MSthresh = s_data.MSthresh(lgcl_mask_time,:,:);
s_data.lgcl_mask_MS = s_data.lgcl_mask_MS(lgcl_mask_time,:,:);
s_data.x_time = s_data.x_time(lgcl_mask_time);

end

function s_data = local_compute_angles(s_data, xchan_idx, ychan_idx)

ntrl = length(s_data.trialinfo);

% number of eyetracked is given by the number of x channels
neye = numel(xchan_idx);

s_data.MS_features = cell(neye, ntrl);
s_data.lastMS = nan(ntrl, 6, neye);
s_data.avgMS = nan(ntrl, 3, neye);
[s_data.T_onoff, s_data.MS_angles, s_data.peak_velMS] = deal(cell(ntrl, neye));

for iEye = 1:neye

    for iTrl = 1:ntrl

        % onset & offset on X axis
        x_onsets = s_data.trial(s_data.lgcl_MS_onset(:,iTrl, iEye), xchan_idx(iEye), iTrl);
        x_offsets = s_data.trial(s_data.lgcl_MS_offset(:,iTrl, iEye), xchan_idx(iEye), iTrl);

        % onset & offset on Y axis
        y_onsets = s_data.trial(s_data.lgcl_MS_onset(:,iTrl, iEye), ychan_idx(iEye), iTrl);
        y_offsets = s_data.trial(s_data.lgcl_MS_offset(:,iTrl, iEye), ychan_idx(iEye), iTrl);

        % difference vectors
        diff_vects = [x_offsets, y_offsets] - [x_onsets, y_onsets];
        
        % peak velocity
        % 1. determine current logical mask
        this_MSs = s_data.lgcl_mask_MS(:,iTrl);
        clusts = bwlabel(this_MSs); lbls_MSs = unique(clusts);
        if any(lbls_MSs>0)
            
            interest_pts = lbls_MSs(lbls_MSs>0);
            max_vel = nan(numel(interest_pts), 1);
            
            for iMS = interest_pts'
                
                this_mask = clusts == iMS;
                this_velocity_segment = s_data.velocity(this_mask, :, iTrl);
                eucl_vel = sqrt(this_velocity_segment(:, 1).^2 + this_velocity_segment(:, 2).^2);
                max_vel(iMS) = max(eucl_vel);
                
            end
            
            s_data.peak_velMS{iTrl} = max_vel;
            
        end
                        
        % compute "Average MS" as the sum of MS vectors, and compute angles.
        % Then store it in a corresponding subfield in s_data.
        res_vect = sum(diff_vects,1);
        if ~isempty(res_vect)     
            angle_avg = atan2(res_vect(2), res_vect(1)); % Y before, x after
            % colord: 1) angle, 2) X, 3) Y
            s_data.avgMS(iTrl, :, iEye) = [angle_avg, res_vect];
        end

        % compute angles
        these_angles = atan2d(diff_vects(:,2), diff_vects(:,1));

        % store time -saccade offset?
        these_Ts = s_data.x_time(s_data.lgcl_MS_offset(:,iTrl, iEye));
        
        % create another subfield specifically for time onsets and offsets
        % for each mMS
        onset_offset = [s_data.x_time(s_data.lgcl_MS_onset(:,iTrl, iEye)), ...
            s_data.x_time(s_data.lgcl_MS_offset(:,iTrl, iEye))];

        % small summary matrix of MS direction and T. only last saccade will be
        % taken into account 
        % colord = 1) Time, 2) x onset, 3) y onset, 4) x offset, 5) y offset, 6) angles
        
        summat = [these_Ts, x_onsets, y_onsets, x_offsets, y_offsets,...
            these_angles]; 

        % attech to the structure
        s_data.MS_features{iEye, iTrl} = summat;
        s_data.T_onoff{iTrl, iEye} = onset_offset;
        s_data.MS_angles{iTrl, iEye} = these_angles;

        if ~isempty(summat)

            s_data.lastMS(iTrl,:, iEye) = summat(end,:);

        else

            s_data.lastMS(iTrl, :, iEye) = nan;

        end

    end

end
end

function s_data = local_select_binocularMS(s_data)
% apply binocular selection of MSs

ntrl = length(s_data.trialinfo);

s_data.bin_MS_angles = cell(ntrl,1);

for iTrl = 1:ntrl
    
    thistrl = s_data.T_onoff(iTrl, :);
    
    if ~any(cellfun(@isempty, thistrl))
        
        l1s = thistrl{1}(:,1);
        l2s = thistrl{1}(:,2);
        r1s = thistrl{2}(:,1);        
        r2s = thistrl{2}(:,2);
        
        lgcl_r2l1 = nan(numel(l1s), numel(r2s)); 
        accr2 = 1;
        for ir2 = r2s'           
            lgcl_r2l1(:, accr2) = ir2>l1s;
            accr2 = accr2+1;
        end
        
        lgcl_l2r1 = nan(numel(l2s), numel(r1s));
        accr1 = 1;
        for ir1 = r1s'           
            lgcl_l2r1(:, accr1) = ir1<l2s;
            accr1 = accr1+1;
        end

        % now determine which MS has happened on both eyes
        swap_logical = lgcl_r2l1 & lgcl_l2r1;
        
        [bin_L_idx, bin_R_idx] = find(swap_logical);
        
        % proceed iff there's one common MS has been found
        if ~isempty(bin_L_idx) && ~isempty(bin_R_idx)
        
            % determine mean angle of the binocular MSs
            left_angles = s_data.MS_angles{iTrl, 1};
            left_angles = repmat(left_angles, 1, numel(r2s));
            left_angles = left_angles(swap_logical);

            right_angles = s_data.MS_angles{iTrl, 2};
            right_angles = repmat(right_angles', numel(l2s), 1);
            right_angles = right_angles(swap_logical);

            % compute circular mean
            bin_angles = circmean_deg([left_angles, right_angles])';
            
            % pipe back the computed angle(s)
            s_data.bin_MS_angles{iTrl} = bin_angles;
        
        end
        
    end
        
end








end

