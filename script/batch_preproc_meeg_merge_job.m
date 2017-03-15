%-----------------------------------------------------------------------
% Job saved on 07-May-2013 14:36:24 by cfg_util (rev $Rev: 4972 $)
% spm SPM - SPM12b (beta)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
matlabbatch{1}.spm.meeg.preproc.merge.D = '<UNDEFINED>';
matlabbatch{1}.spm.meeg.preproc.merge.rule.file = '.*';
matlabbatch{1}.spm.meeg.preproc.merge.rule.labelorg = '.*';
matlabbatch{1}.spm.meeg.preproc.merge.rule.labelnew = '#labelorg#';
matlabbatch{1}.spm.meeg.preproc.merge.prefix = 'c';
matlabbatch{2}.spm.meeg.other.delete.D = '<UNDEFINED>';
matlabbatch{3}.spm.meeg.preproc.prepare.D(1) = cfg_dep('Merging: Merged Datafile', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{3}.spm.meeg.preproc.prepare.task{1}.avref.fname = {'avref_montage.mat'};
matlabbatch{4}.spm.meeg.preproc.montage.D(1) = cfg_dep('Prepare: Prepared Datafile', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{4}.spm.meeg.preproc.montage.mode.write.montspec.montage.montagefile(1) = cfg_dep('Prepare: Average reference montage', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','avrefname'));
matlabbatch{4}.spm.meeg.preproc.montage.mode.write.montspec.montage.keepothers = 1;
matlabbatch{4}.spm.meeg.preproc.montage.mode.write.blocksize = 655360;
matlabbatch{4}.spm.meeg.preproc.montage.mode.write.prefix = 'M';
matlabbatch{5}.spm.meeg.preproc.prepare.D(1) = cfg_dep('Montage: Montaged Datafile', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));
matlabbatch{5}.spm.meeg.preproc.prepare.task{1}.sortconditions.label = {
                                                                        'Famous'
                                                                        'Unfamiliar'
                                                                        'Scrambled'
                                                                        }';
matlabbatch{6}.spm.meeg.other.delete.D(1) = cfg_dep('Merging: Merged Datafile', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','Dfname'));


