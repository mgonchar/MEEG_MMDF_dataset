function eval_MSMM(run_preproc, use_empty_room)

clc;

if nargin < 1
    run_preproc = true;
end

if nargin < 2
    use_empty_room = true;
end

%% define globals

subjects = {
    'sub001';
    'sub002';
    %'sub003';
    %'sub004';
    %'sub005';
    %'sub006';
    %'sub007';
    %'sub008';
    %'sub009';
    %'sub010';
    %'sub011';
    %'sub012';
    %'sub013';
    %'sub014';
    %'sub015';
    %'sub016';
    %'sub017';
    %'sub018';
    %'sub019'
};

n_subjects = length(subjects);

% channel to extract trials
stimulus_channel = 'STI101';

% hardcoded delay in ms
stimulus_delay = 34;

% epoch offsets in ms
onset  = 500;
offset = 1200;
prestimlulus_baseline = -510;

% sampling rate in Hz
sampling_rate = 1100;
HzToMs_ratio  = sampling_rate / 1000;

% mask to remap event codes to classes
event_codes = [5 6 7 13 14 15 17 18 19];
event_type_mask = containers.Map(event_codes, {'Famous', 'Famous', 'Famous', 'Unfamiliar', 'Unfamiliar', 'Unfamiliar', 'Scrambled', 'Scrambled', 'Scrambled'});

% path to BrainStorm database
dirBS_db = 'D:/science/Brain/MSMM/dataset/brainstorm_db/Protocol01/';

% path to unzipped dataset
dir_dataset = 'D:/science/Brain/MSMM/dataset/ds117';
dir_metha   = 'D:/science/Brain/MSMM/dataset/ds117_R0.1.1_metadata';

% path to output
outpth = 'D:/science/Brain/MSMM/output';

% path to SPM scripts
scrpth = 'D:/science/Brain/MSMM/script';

% empty room records mapping
all_sbj = {'sub001'; 'sub002'; 'sub003'; 'sub004'; 'sub005'; 'sub006'; 'sub007'; 'sub008'; 'sub009'; 'sub010'; 'sub011'; 'sub012'; 'sub013'; 'sub014'; 'sub015'; 'sub016'; 'sub017'; 'sub018'; 'sub019'};
recs    = [repmat({'090430_raw_st.fif'},1,16), repmat({'090707_raw_st.fif'},1,3)];
empty_room_mask = containers.Map(all_sbj,recs);

%% create folders structure if required
if ~exist(outpth, 'dir')
    eval(sprintf('!mkdir %s',outpth)); 
end
    
for s = 1:n_subjects
    fullsubdir = fullfile(outpth,subjects{s});
    if ~exist(fullsubdir, 'dir')
    	eval(sprintf('!mkdir %s',fullsubdir)); 
    end
end
clear fullsubdir;

%% preprocess the data
if run_preproc
    for s = 1:n_subjects
        %% Convert & epoch, prepare, downsample, baseline-correct each run

        % get number of runs from file structure
        MEG_path = [dir_dataset,'/',subjects{s},'/MEG/'];
        f = dir(MEG_path);
        f = regexpi({f.name},'run_\d+_sss.fif','match');
        f = [f{:}];
        nrun = length(f);
        clear f;

        jobfile = {fullfile(scrpth,'batch_preproc_meeg_convert_job.m')};
        jobs    = repmat(jobfile, 1, nrun);
        n = 1;
        inputs  = cell(nrun*3, 1);
        for r = 1:nrun
            %% get bad channels from logfile, if not done yet
            if ~exist([MEG_path,sprintf('run_%02d_bdch.mat',r)], 'file')
    %            bad_channels = [];
    %            
    %            fid = fopen([MEG_path,sprintf('run_%02d_sss_log.txt',r)]);
    %            tline = fgets(fid);
    %            
    %            % grep a string pattern
    %            if ~isempty(strfind(tline, 'Static bad channels'))
    %                tmp = strsplit(tline, ':');
    %                bad_channels = union(bad_channels, sscanf(tmp{2}, '%d'));
    %            end
    %            flose(fid);
    %            clear fid tline tmp;

                label = {};  
                save([MEG_path,sprintf('run_%02d_bdch.mat',r)], 'label');
            end

            %% Get events from events channel, if not done yet
            if ~exist([MEG_path,sprintf('run_%02d_trldf.mat',r)], 'file')
                raw = fiff_setup_read_raw([MEG_path,sprintf('run_%02d_sss.fif',r)]);

                picks = fiff_pick_types(raw.info,false,false,false,{stimulus_channel},raw.info.bads);

                [ data, ~ ] = fiff_read_raw_segment(raw,raw.first_samp,raw.last_samp,picks);

                is_event = zeros(size(data));
                for code = event_codes
                    is_event = is_event | data == code;
                end

                exctracted_events     = data(is_event);
                exctracted_events_idx = find(is_event);
                final_events      = exctracted_events_idx(1);
                conditionlabels   = {event_type_mask(data(exctracted_events_idx(1)))};

                for i = 1:length(exctracted_events)-1
                    if exctracted_events(i) ~= exctracted_events(i+1) || exctracted_events(i) == exctracted_events(i+1) && exctracted_events_idx(i+1) - exctracted_events_idx(i) > 1
                        final_events(end+1)    = exctracted_events_idx(i+1);
                        conditionlabels(end+1) = {event_type_mask(data(exctracted_events_idx(i+1)))};
                    end
                end
                trl = [(final_events' - onset*HzToMs_ratio) (final_events' + offset*HzToMs_ratio) repmat(prestimlulus_baseline, length(final_events), 1)];
                save([MEG_path,sprintf('run_%02d_trldf.mat',r)], 'trl', 'conditionlabels');

                clear raw picks data;
            end

            %% fill SPM batch for preproc
            inputs{n  ,1} = cellstr(fullfile(MEG_path,sprintf('run_%02d_sss.fif',r)));
            inputs{n+1,1} = cellstr(fullfile(MEG_path,sprintf('run_%02d_trldf.mat',r)));
            inputs{n+2,1} = cellstr(fullfile(MEG_path,sprintf('run_%02d_bdch.mat',r)));
            n = n + 3;
        end

        %% evaluate SPM batch
        % change current folder for redirecting the output
        currFolder = cd(fullfile(outpth,subjects{s}));

        spm_jobman('serial', jobs, '', inputs{:});

        %% Concatenate runs and montage (reref EEG) 
        jobfile = {fullfile(scrpth,'batch_preproc_meeg_merge_job.m')};   
        inputs  = cell(3, 1);
        inputs{1} = cellstr(spm_select('FPList',fullfile(outpth,subjects{s}),'^bdspmeeg.*\.mat$'));
        inputs{2} = cellstr(spm_select('FPList',fullfile(outpth,subjects{s}),'^bdspmeeg.*\.mat$'));  % (For deletion)  
        spm_jobman('serial', jobfile, '', inputs{:});

        %% or ERP/ERF: crop to -100 to +800, detect artifacts (blinks) by thresholding EOG, average over trials and create contrasts of conditions
        jobfile = {fullfile(scrpth,'batch_preproc_meeg_erp_job.m')};
        inputs  = cell(1);
        inputs{1} = cellstr(spm_select('FPList',fullfile(outpth,subjects{s}),'^Mcbdspmeeg.*\.mat$'));
        spm_jobman('serial', jobfile, '', inputs{:})

        % return back to script folder
        cd(currFolder);
    end
end

%% calculate inverse
%     Let's scale all of the individual meshes to the default one, with operator W, i.e.
% 
%     Y = L*J
%     Y = L*W_-1*W*J, where W: individual->default
%     Y = L'*J', where L' = L*W_-1, J' = W*J
% 
%     J' = M'*Y
%     J = W_-1*M'*Y

% Pre-alloc globals
Lp      = cell(n_subjects,1);
Lp_norm = cell(n_subjects,1);
Y       = cell(n_subjects,5); % (n_subjects, D.nconditions)
Y_norm  = cell(n_subjects,5); % (n_subjects, D.nconditions)

srcSurfFile = strcat(dirBS_db,'anat/@default_subject/tess_cortex_pial_low.mat');

% apply Multiple Factor Analysis (MFA) to reduce spatial redundancy,
% i.e:
% 
% step 1: get PCA for each modality of each subject
% step 2: normalize original data by first singular value
% step 3: concatenate normalized data along dimension that is not of
%         interest
% step 4: get PCA of extended data
% step 5: apply PCA from step 4 to original data (independently for each
%         set)
    
for s = 1:n_subjects
    
    % load brainstorm mesh
    sb = subjects{s};
    sb = [sb(1:3), 'j', sb(4:end)];
    if exist(strcat(dirBS_db,'data/',subjects{s},'/',sb,'_run_01_sss/headmodel_surf_os_meg.mat'),'file')
        load(strcat(dirBS_db,'data/',subjects{s},'/',sb,'_run_01_sss/headmodel_surf_os_meg.mat'));
    else
        load(strcat(dirBS_db,'data/',subjects{s},'/run_01_sss/headmodel_surf_os_meg.mat'));
    end
    
    % load gain matrix (forward operator)
    Lp{s} = bst_gain_orient(Gain, GridOrient);
    
    % Brainstorm indexes mismatches with SPM ones. Also there are no bad
    % channels for MEG, so just omit them here.
    if exist(strcat(dirBS_db,'data/',subjects{s},'/',sb,'_run_01_sss/channel_vectorview306.mat'), 'file')
        load(strcat(dirBS_db,'data/',subjects{s},'/',sb,'_run_01_sss/channel_vectorview306.mat'));
    else
        if exist(strcat(dirBS_db,'data/',subjects{s},'/run_01_sss/channel_vectorview306.mat'), 'file')
            load(strcat(dirBS_db,'data/',subjects{s},'/run_01_sss/channel_vectorview306.mat'));
        else
            load(strcat(dirBS_db,'data/',subjects{s},'/run_01_sss/channel_vectorview306_acc1.mat'));
        end
    end
    Ic_bst = false(size(Channel));
    for i = 1:length(Channel)
        Ic_bst(i) = strcmp(Channel(i).Type, 'MEG GRAD');
    end

    % use only MEGPLANAR
    Lp{s} = Lp{s}(Ic_bst,:);
    
    % inverse interpolation - instead of real inverse I use forward one but
    % from dest mesh to source one
    destSurfFile = fullfile(dirBS_db,'anat',subjects{s},'tess_cortex_pial_low.mat');
    Wmat = my_interpolation(srcSurfFile, destSurfFile, 8);
    
    Lp{s} = Lp{s}*Wmat;
    [N_chan, N_vert] = size(Lp{s});
    
    % eliminate low SNR spatial modes (reduce sensor space)
    [~,S,~] = svd(cov(Lp{s}')); % step 1
    Lp_norm{s} = Lp{s} / S(1);  % step 2
end

clear Wmat S sb;

tmp = zeros(N_chan, n_subjects*N_vert);
for s = 1:n_subjects
    before = (s-1)*N_vert;
    tmp(:, before + 1 : before + N_vert) = Lp_norm{s}; % step 3
end

[coeff,~,~,~,explained,~] = pca(tmp'); % step 4
N_spat = find(cumsum(explained) > 99.9, 1, 'first');

U = coeff(:,1:N_spat); % spatial projector
clear tmp Lp_norm coeff explained;

% now get to step 5 and prepare data for temporal projection

% create a long-data array for L and noise covariance
%
% UL = blckdiag(U * Lp); 
% Qe = blckdiag(U' * Qep * U);
Qe = zeros(n_subjects*N_spat);
UL = zeros(n_subjects*N_spat, n_subjects*N_vert); 

for s = 1:n_subjects
    
    % load SPM processed data
    D = spm_eeg_load(fullfile(outpth,subjects{s},'wmapMcbdspmeeg_run_01_sss.mat'));
    Ic_spm = setdiff(D.indchantype('MEGPLANAR'), D.badchannels);
    
    % apply MFA to reduce temporal redundancy
    for j = 1:D.nconditions
        Y{s,j}  = U'*D(Ic_spm,:,j);  % project out low SNR spatial modes
        [~,S,~] = svd(cov(Y{s,j}));  % step 1
        Y_norm{s,j} = Y{s,j} / S(1); % step 2
    end
    
    % prepare block diagonal noise covariance matrix
    if use_empty_room
        % save calculated covariance matrix to speed up calculation
        name = empty_room_mask(subjects{s});
        if ~exist(['tmp_',name(1:end-4),'_cov.mat'], 'file')
            
            raw = fiff_setup_read_raw(fullfile(dir_metha,'emptyroom',name));
            picks = fiff_pick_types(raw.info,true,false,false,[],raw.info.bads);
            
            % pick only MEG PLANAR GRAD
            pick_mask = true(size(picks));
            for i = 1:length(picks)
                pick_mask(i) = raw.info.chs(picks(i)).coil_type ~= 3022;
            end
            [ data, ~ ] = fiff_read_raw_segment(raw,raw.first_samp,raw.last_samp,picks(pick_mask));
            noise_cov = cov(data'); clear raw picks data;
            
            save(['tmp_',name(1:end-4),'_cov.mat'], 'noise_cov');
        else
            load(['tmp_',name(1:end-4),'_cov.mat']);
        end
    else
        % fallback is to use eye noise covaraince prior
        noise_cov = eye(N_chan);
    end
    
    before = (s-1)*N_spat;
    Qe(before + 1 : before + N_spat, before  + 1 : before  + N_spat) = U'*noise_cov*U; clear noise_cov;
    
    % block diagonal Leadfield matrix (with spatiat projector)
    before2 = (s-1)*N_vert;
    UL(before + 1 : before + N_spat, before2 + 1 : before2 + N_vert) = U'*Lp{s}; % step 5
end

clear S Lp;

tmp = zeros(D.nconditions*n_subjects*N_spat, size(Y_norm{1,1},2));

for s = 1:n_subjects
    for j = 1:D.nconditions
        before = (s-1)*D.nconditions * N_spat  + (j-1)*N_spat;
        tmp(before + 1 : before + N_spat, :) = Y_norm{s,j}; % step 3
    end
end

[coeff,~,~,~,explained,~] = pca(tmp); % step 4
N_temp = find(cumsum(explained) > 99.9, 1, 'first');

T = coeff(:,1:N_temp); % temporal projector
% N_temp_orig = size(T,1);

clear tmp coeff explained Y_norm;

% create a long-data array of Y as column vector of individual measurement,
% with common temporal projector T to reduce temporal redundancy, i.e:
%
% UY = ( ( Y_1,1 * T ) ( Y_1,2 * T ) ... (Y_n_subj,n_cond * T))' 
%        (subj1 cond1) (subj1 cond2) ... (last_subj last_cond)                    
UY = zeros(N_spat*n_subjects, D.nconditions*N_temp);
for s = 1:n_subjects
    before  = (s-1) * N_spat;
    for j = 1:D.nconditions
        before2 = (j-1)*N_temp; 
        UY(before + 1 : before + N_spat, before2 + 1 : before2 + N_temp) = Y{s,j}*T; % step 5
    end
end
clear Y;

% make smoothing kernel
load(srcSurfFile);

% contingency matrix
Cm = spm_mesh_distmtx(struct('vertices',Vertices,'faces',Faces),0);
% smoothing weights across subjects' neighbours
y = [1 0.8 0.37 0.2];

iter_idx = 2:3;
AA  = cell(max(iter_idx)+1, 1);

% find subsets of certain distance
AA{1} = eye(N_vert);
AA{2} = full(Cm);
B = AA{1} + AA{2};
for i = iter_idx
    Bn = zeros(N_vert);
    Bn(Cm^i ~= 0) = 1;
    AA{i+1} = full(Bn - B);
    B = Bn;
end
clear B Bn Cm;

QG = zeros(N_vert);
for i = 1:max(iter_idx)+1
    QG = QG + AA{i}*y(i);
end
K = sparse(QG(:,:));

clear AA y QG iter_idx;

% use calculate_distance_file code
%load('distance5002.mat');

Vs = speye(n_subjects*N_vert); % Initialization of ROI vertices matrix

% template for source-noise covariance
psQ1 = kron(ones(n_subjects),  K);
psQ2 = kron(speye(n_subjects), K);
% psQ2 = kron(speye(Nl),speye(Nd));

%% Per-condition inverse (GALA)
%for j = 1:D.nconditions
%    before = (j-1)*N_spat;
%    UY_c = UY(before + 1 : before + N_spat, :);
%    YY = cov(UY_c');%UY*UY';
% 
%    % first ROI covariance matrix - strong correlation between subjects
%    sQ1 = Vs*psQ1*Vs;
%    Q1 = full(UL*sQ1*UL');
% 
%    % second ROI covariance matrix - subjects specific activity
%    sQ2 = Vs*psQ2*Vs;
%    Q2 = full(UL*sQ2*UL');
% 
%    Qs = {Q1 Q2}; clear Q1 Q2;
%
%    % noise covariance component
%    Qn{1} = Qe; % it's the first one for channel noise
% 
%    [Cy,h,Ph,F] = spm_reml_sc(YY,[],[Qn Qs],1,-4,16);
% 
%    DD = h(end-1)*sQ1 + h(end)*sQ2;
% 
%    Cy = full(Cy);    
%    M = DD*UL'/Cy;
% 
%    J = M*UY_c * T';   
%end

%% All condition inverse (GALA)
    YY = (D.nconditions*N_temp-1)*cov(UY');

    % first ROI covariance matrix - strong correlation between subjects
    sQ1 = Vs*psQ1*Vs;
    Q1 = full(UL*sQ1*UL');

    % second ROI covariance matrix - subjects specific activity
    sQ2 = Vs*psQ2*Vs;
    Q2 = full(UL*sQ2*UL');

    Qs = {Q1 Q2}; clear Q1 Q2;
    
    % noise covariance component
    Qn = {Qe}; % it's the first one for channel noise

    %[Cy,h,Ph,F] = spm_reml_sc(YY,[],[Qn Qs],1,-4,16);
    [Cy,h,Ph,F] = spm_reml_sc(YY, [], [Qn Qs], D.nconditions*N_temp);%,-4,16);

    DD = h(end-1)*sQ1 + h(end)*sQ2; clear sQ1 sQ2;

    %Cy = full(Cy);    
    M = DD*UL'/Cy;

    J = M*UY;  
    
    J_gala = cell(n_subjects, D.nconditions);
    for s = 1:n_subjects
        for j = 1:D.nconditions
            J_gala{s,j} = J((s-1)*N_vert + 1 : s*N_vert, (j-1)*N_temp + 1 : j*N_temp) * T';
        end
    end
    
%% MNE

    YY = (D.nconditions*N_temp-1)*cov(UY');

%     % first ROI covariance matrix - strong correlation between subjects
%     sQ1 = Vs*psQ1*Vs;
%     Q1 = full(UL*sQ1*UL');
% 
%     % second ROI covariance matrix - subjects specific activity
%     sQ2 = Vs*psQ2*Vs;
%     Q2 = full(UL*sQ2*UL');
% 
%     Qs = {Q1 Q2}; clear Q1 Q2;
%     
%     % noise covariance component
%     Qn = {Qe}; % it's the first one for channel noise

    %[Cy,h,Ph,F] = spm_reml_sc(YY,[],[Qn Qs],1,-4,16);
    [Cy,h,Ph,F] = spm_reml_sc(YY,[],{UL*UL'},D.nconditions*N_temp);%,-4,16);

    %DD = h(end-1)*sQ1 + h(end)*sQ2; clear sQ1 sQ2;
    %DD = h(end)*speye(n_subjects*N_vert);

    %Cy = full(Cy);    
    %M = DD*UL'/Cy;
    M = h(end)*UL'/Cy;

    J = M*UY;  
    
    J_mne = cell(n_subjects, D.nconditions);
    for s = 1:n_subjects
        for j = 1:D.nconditions
            J_mne{s, j} = J((s-1)*N_vert + 1 : s*N_vert, (j-1)*N_temp + 1 : j*N_temp) * T';
        end
    end
    
%% verify

% remove temps
if exist('tmp_090430_raw_st_cov.mat', 'file')
    delete('tmp_090430_raw_st_cov.mat');
end
if exist('tmp_090707_raw_st_cov.mat', 'file')
    delete('tmp_090707_raw_st_cov.mat');
end
end

