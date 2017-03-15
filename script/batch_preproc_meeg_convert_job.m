%-----------------------------------------------------------------------
% Job saved on 03-May-2013 21:46:28 by cfg_util (rev $Rev: 4972 $)
% spm SPM - SPM12b (beta)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.meeg.convert.dataset = '<UNDEFINED>';
matlabbatch{1}.spm.meeg.convert.mode.epoched.trlfile = '<UNDEFINED>';
matlabbatch{1}.spm.meeg.convert.channels{1}.type = 'EEG';
matlabbatch{1}.spm.meeg.convert.channels{2}.type = 'MEGMAG';
matlabbatch{1}.spm.meeg.convert.channels{3}.type = 'MEGPLANAR';
matlabbatch{1}.spm.meeg.convert.outfile = '';
matlabbatch{1}.spm.meeg.convert.eventpadding = 0;
matlabbatch{1}.spm.meeg.convert.blocksize = 3276800;
matlabbatch{1}.spm.meeg.convert.checkboundary = 1;
matlabbatch{1}.spm.meeg.convert.saveorigheader = 0;
matlabbatch{1}.spm.meeg.convert.inputformat = 'autodetect';
matlabbatch{2}.spm.meeg.preproc.prepare.D(1) = cfg_dep('Conversion: Converted Datafile', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{2}.spm.meeg.preproc.prepare.task{1}.settype.channels{1}.chan = 'EEG061';
matlabbatch{2}.spm.meeg.preproc.prepare.task{1}.settype.channels{2}.chan = 'EEG062';
matlabbatch{2}.spm.meeg.preproc.prepare.task{1}.settype.newtype = 'EOG';
matlabbatch{2}.spm.meeg.preproc.prepare.task{2}.settype.channels{1}.chan = 'EEG063';
matlabbatch{2}.spm.meeg.preproc.prepare.task{2}.settype.newtype = 'ECG';
matlabbatch{2}.spm.meeg.preproc.prepare.task{3}.settype.channels{1}.chan = 'EEG064';
matlabbatch{2}.spm.meeg.preproc.prepare.task{3}.settype.newtype = 'Other';
matlabbatch{2}.spm.meeg.preproc.prepare.task{4}.setbadchan.channels{1}.chanfile = '<UNDEFINED>';
matlabbatch{2}.spm.meeg.preproc.prepare.task{4}.setbadchan.status = 1;
matlabbatch{3}.spm.meeg.preproc.downsample.D(1) = cfg_dep('Prepare: Prepared Datafile', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{3}.spm.meeg.preproc.downsample.fsample_new = 200;
matlabbatch{3}.spm.meeg.preproc.downsample.prefix = 'd';
matlabbatch{4}.spm.meeg.preproc.bc.D(1) = cfg_dep('Downsampling: Downsampled Datafile', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{4}.spm.meeg.preproc.bc.timewin = [-100 0];
matlabbatch{4}.spm.meeg.preproc.bc.prefix = 'b';
matlabbatch{5}.spm.meeg.other.delete.D(1) = cfg_dep('Prepare: Prepared Datafile', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{6}.spm.meeg.other.delete.D(1) = cfg_dep('Downsampling: Downsampled Datafile', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
