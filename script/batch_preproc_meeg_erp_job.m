%-----------------------------------------------------------------------
% Job saved on 07-May-2013 15:05:02 by cfg_util (rev $Rev: 4972 $)
% spm SPM - SPM12b (beta)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.meeg.preproc.crop.D = '<UNDEFINED>';
matlabbatch{1}.spm.meeg.preproc.crop.timewin = [-100 800];
matlabbatch{1}.spm.meeg.preproc.crop.freqwin = [-Inf Inf];
matlabbatch{1}.spm.meeg.preproc.crop.channels{1}.all = 'all';
matlabbatch{1}.spm.meeg.preproc.crop.prefix = 'p';
matlabbatch{2}.spm.meeg.preproc.artefact.D(1) = cfg_dep('Crop: Cropped M/EEG datafile', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{2}.spm.meeg.preproc.artefact.badchanthresh = 0.2;
matlabbatch{2}.spm.meeg.preproc.artefact.methods.channels{1}.type = 'EOG';
matlabbatch{2}.spm.meeg.preproc.artefact.methods.fun.threshchan.threshold = 200;
matlabbatch{2}.spm.meeg.preproc.artefact.prefix = 'a';
%matlabbatch{3}.spm.meeg.preproc.combineplanar.D(1) = cfg_dep('Artefact detection: Artefact-detected Datafile', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
%matlabbatch{3}.spm.meeg.preproc.combineplanar.mode = 'replace';
%matlabbatch{3}.spm.meeg.preproc.combineplanar.prefix = 'P';
matlabbatch{3}.spm.meeg.other.delete.D(1) = cfg_dep('Crop: Cropped M/EEG datafile', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
%matlabbatch{5}.spm.meeg.averaging.average.D(1) = cfg_dep('Combine planar: Planar-combined MEG datafile', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{4}.spm.meeg.averaging.average.D(1) = cfg_dep('Artefact detection: Artefact-detected Datafile', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{4}.spm.meeg.averaging.average.userobust.standard = false;
matlabbatch{4}.spm.meeg.averaging.average.plv = false;
matlabbatch{4}.spm.meeg.averaging.average.prefix = 'm';
matlabbatch{5}.spm.meeg.averaging.contrast.D(1) = cfg_dep('Averaging: Averaged Datafile', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(1).c = [1 0 0];
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(1).label = 'Famous';
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(2).c = [0 1 0];
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(2).label = 'Unfamiliar';
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(3).c = [0 0 1];
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(3).label = 'Scrambled';
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(4).c = [0.5 0.5 -1];
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(4).label = 'Faces - Scrambled';
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(5).c = [1 -1 0];
matlabbatch{5}.spm.meeg.averaging.contrast.contrast(5).label = 'Famous - Unfamiliar';
% matlabbatch{6}.spm.meeg.averaging.contrast.contrast(6).c = [0.5 0.5 0];
% matlabbatch{6}.spm.meeg.averaging.contrast.contrast(6).label = 'Faces';
% matlabbatch{6}.spm.meeg.averaging.contrast.contrast(3).c = [0 1 -1];
% matlabbatch{6}.spm.meeg.averaging.contrast.contrast(3).label = 'Unfamiliar - Scrambled';
% matlabbatch{6}.spm.meeg.averaging.contrast.contrast(4).c = [0.333333333333333 0.333333333333333 0.333333333333333];
% matlabbatch{6}.spm.meeg.averaging.contrast.contrast(4).label = 'All';
matlabbatch{5}.spm.meeg.averaging.contrast.weighted = 1;
matlabbatch{5}.spm.meeg.averaging.contrast.prefix = 'w';
matlabbatch{6}.spm.meeg.other.delete.D(1) = cfg_dep('Averaging: Averaged Datafile', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
