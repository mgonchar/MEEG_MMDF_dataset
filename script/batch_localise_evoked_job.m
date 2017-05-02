%-----------------------------------------------------------------------
% Job saved on 10-May-2013 09:17:20 by cfg_util (rev $Rev: 4972 $)
% spm SPM - SPM12b (beta)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------

matlabbatch{1}.spm.meeg.source.invert.D = '<UNDEFINED>';
matlabbatch{1}.spm.meeg.source.invert.val = 1;
matlabbatch{1}.spm.meeg.source.invert.whatconditions.all = 1;
matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.invtype = 'GS';
matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.woi = [-100 800];
matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.foi = [0 256];
matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.hanning = 1;
matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.priors.priorsmask = '<UNDEFINED>'; 
matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.priors.space = 1;
matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.restrict.locs = zeros(0, 3);
matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.restrict.radius = 32;
matlabbatch{1}.spm.meeg.source.invert.modality = {'All'};

matlabbatch{2}.spm.meeg.source.invert.D = '<UNDEFINED>'; %cfg_dep('Source inversion: M/EEG dataset(s) after imaging source reconstruction', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','D'));
matlabbatch{2}.spm.meeg.source.invert.val = 2;
matlabbatch{2}.spm.meeg.source.invert.whatconditions.all = 1;
matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.invtype = 'IID';
matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.woi = [-100 800];
matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.foi = [0 256];
matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.hanning = 1;
matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.priors.priorsmask = '<UNDEFINED>'; 
matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.priors.space = 1;
matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.restrict.locs = zeros(0, 3);
matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.restrict.radius = 32;
matlabbatch{2}.spm.meeg.source.invert.modality = {'All'};

matlabbatch{3}.spm.meeg.source.results.D(1) = cfg_dep('Source inversion: M/EEG dataset(s) after imaging source reconstruction', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','D'));
matlabbatch{3}.spm.meeg.source.results.val = 1;
matlabbatch{3}.spm.meeg.source.results.woi = [100 250];
matlabbatch{3}.spm.meeg.source.results.foi = [10 20];
matlabbatch{3}.spm.meeg.source.results.ctype = 'evoked';
matlabbatch{3}.spm.meeg.source.results.space = 1;
%matlabbatch{3}.spm.meeg.source.results.format = 'image';
matlabbatch{3}.spm.meeg.source.results.format = 'mesh';
matlabbatch{3}.spm.meeg.source.results.smoothing = 8;

matlabbatch{4}.spm.meeg.source.results.D(1) = cfg_dep('Source inversion: M/EEG dataset(s) after imaging source reconstruction', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','D'));
matlabbatch{4}.spm.meeg.source.results.val = 2;
matlabbatch{4}.spm.meeg.source.results.woi = [100 250];
matlabbatch{4}.spm.meeg.source.results.foi = [10 20];
matlabbatch{4}.spm.meeg.source.results.ctype = 'evoked';
matlabbatch{4}.spm.meeg.source.results.space = 1;
%matlabbatch{4}.spm.meeg.source.results.format = 'image';
matlabbatch{4}.spm.meeg.source.results.format = 'mesh';
matlabbatch{4}.spm.meeg.source.results.smoothing = 8;

% matlabbatch{1}.spm.meeg.source.invert.D = '<UNDEFINED>';
% matlabbatch{1}.spm.meeg.source.invert.val = 1;
% matlabbatch{1}.spm.meeg.source.invert.whatconditions.all = 1;
% matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.invtype = 'GS';
% matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.woi = [-100 800];
% matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.foi = [0 256];
% matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.hanning = 1;
% matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.priors.priorsmask = {''};
% matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.priors.space = 1;
% matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.restrict.locs = zeros(0, 3);
% matlabbatch{1}.spm.meeg.source.invert.isstandard.custom.restrict.radius = 32;
% matlabbatch{1}.spm.meeg.source.invert.modality = {'EEG'};
% 
% matlabbatch{2}.spm.meeg.source.invert.D = cfg_dep('Source inversion: M/EEG dataset(s) after imaging source reconstruction', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','D'));
% matlabbatch{2}.spm.meeg.source.invert.val = 2;
% matlabbatch{2}.spm.meeg.source.invert.whatconditions.all = 1;
% matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.invtype = 'GS';
% matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.woi = [-100 800];
% matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.foi = [0 256];
% matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.hanning = 1;
% matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.priors.priorsmask = {''};
% matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.priors.space = 1;
% matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.restrict.locs = zeros(0, 3);
% matlabbatch{2}.spm.meeg.source.invert.isstandard.custom.restrict.radius = 32;
% matlabbatch{2}.spm.meeg.source.invert.modality = {'MEG'};
% 
% matlabbatch{3}.spm.meeg.source.invert.D = cfg_dep('Source inversion: M/EEG dataset(s) after imaging source reconstruction', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','D'));
% matlabbatch{3}.spm.meeg.source.invert.val = 3;
% matlabbatch{3}.spm.meeg.source.invert.whatconditions.all = 1;
% matlabbatch{3}.spm.meeg.source.invert.isstandard.custom.invtype = 'GS';
% matlabbatch{3}.spm.meeg.source.invert.isstandard.custom.woi = [-100 800];
% matlabbatch{3}.spm.meeg.source.invert.isstandard.custom.foi = [0 256];
% matlabbatch{3}.spm.meeg.source.invert.isstandard.custom.hanning = 1;
% matlabbatch{3}.spm.meeg.source.invert.isstandard.custom.priors.priorsmask = {''};
% matlabbatch{3}.spm.meeg.source.invert.isstandard.custom.priors.space = 1;
% matlabbatch{3}.spm.meeg.source.invert.isstandard.custom.restrict.locs = zeros(0, 3);
% matlabbatch{3}.spm.meeg.source.invert.isstandard.custom.restrict.radius = 32;
% matlabbatch{3}.spm.meeg.source.invert.modality = {'MEGPLANAR'};
% 
