% clear all;
clc;
path_to_db_anat = 'D:\science\Brain\MSMM\dataset\brainstorm_db\Protocol01\anat';
data_to_display = 'D:\science\Brain\MSMM\script\J_all_nbh_off.mat';
atlas_to_use = 'Destrieux'; % 148 areas atlas

% HOWTO: load whatever to variable you want
 tt = whos('-file',data_to_display);
%  name_to_load = tt(2).name;
%  
%  J = load(data_to_display, name_to_load);
%  J = J.(name_to_load);

% J = load(data_to_display);
% J_gala = J.(tt(1).name);
% J_mne  = J.(tt(2).name);
% clear J;
% 
% n_cond = size(J_mne,2);

% HOWTO: count subjects
f = dir(path_to_db_anat);
f = regexpi({f.name},'sub\d+','match');
sbj_list = [f{:}]; % here are the subjects name
n_subj = length(sbj_list);

clrs = ['y' 'm' 'c' 'r' 'g' 'b' 'w' 'k'];
leg = {'Famous', 'Unfamiliar', 'Scrambled', 'Faces - Scrambled', 'Famous - Unfamiliar'};

% change here to adjust time-window
% time_window = 1:length(J_mne{1,1}(1,:));

% Average across subjects
% J_mne_avg  = cell(n_cond,1);
% J_gala_avg = cell(n_cond,1);
% 
% for cnd = 1:n_cond
%     
%     J_mne_avg{cnd}  = J_mne{cnd,1};
%     J_gala_avg{cnd} = J_gala{cnd,1};
%     
%     for s = 2:n_subj
%         J_mne_avg{cnd}  = J_mne{s, cnd};
%         J_gala_avg{cnd} = J_gala{s, cnd};
%     end
% end
% clear J_mne J_gala;

%for s = 1:n_subj
s = 1;
    nm = fullfile(path_to_db_anat, sbj_list(s), 'tess_cortex_pial_low.mat');
    load(nm{1});

    for i = 1:length(Atlas)
        if strcmp(Atlas(i).Name, atlas_to_use)
            atlas_idx = i;
            break;
        end
    end
    
    % Assuming left region always going first
    
    % For all regions in Atlas
    for i = 1:2:length(Atlas(atlas_idx).Scouts)-1
        % Full screen figure
        figure('units','normalized','outerposition',[0 0 1 1]);
        
        % LEFT hemisphere
        subplot(1,2,1);
        hold on;
        
        % For each subject
        %for s = 1:n_subj
            % For each condition
            for cnd = 1:n_cond
                % Plot the seed timecource in time_window 
                %plot(J{s,cnd}(Atlas(atlas_idx).Scouts(i).Seed, time_window), clrs(cnd));
                plot(mean(J_mne_avg{cnd}(Atlas(atlas_idx).Scouts(i).Vertices, time_window),1), clrs(cnd));
                xlim([min(time_window) max(time_window)]);
            end
            
            % Append legend after first iteration
            %if s == 1
                legend(leg);
            %end
            
            for cnd = 1:n_cond
                plot(mean(J_gala_avg{cnd}(Atlas(atlas_idx).Scouts(i).Vertices, time_window),1), clrs(cnd), 'LineWidth', 2)
                xlim([min(time_window) max(time_window)]);
            end
        %end
        % Add title (current region)
        title(Atlas(atlas_idx).Scouts(i).Label,'Interpreter','none', 'FontSize', 20);
        hold off;

        % Same for RIGHT hemisphere
        subplot(1,2,2);
        hold on;
        %for s = 1:n_subj
            for cnd = 1:n_cond
                %plot(J{s,cnd}(Atlas(atlas_idx).Scouts(i+1).Seed, time_window), clrs(cnd));
                plot(mean(J_mne_avg{cnd}(Atlas(atlas_idx).Scouts(i+1).Vertices, time_window),1), clrs(cnd));
                xlim([min(time_window) max(time_window)]);
            end
            %if s == 1
                legend(leg);
            %end
            
            for cnd = 1:n_cond
                plot(mean(J_gala_avg{cnd}(Atlas(atlas_idx).Scouts(i+1).Vertices, time_window),1), clrs(cnd), 'LineWidth', 2)
                xlim([min(time_window) max(time_window)]);
            end
        %end
        title(Atlas(atlas_idx).Scouts(i+1).Label,'Interpreter','none', 'FontSize', 20);
        hold off;
        
        % Get rid of useless space
        tightfig;
        % Maximize the figure
        maximize;
        
        % force the figure to the report (if not, you have to keep it open till the end of capturing)
        snapnow;
        % close the figure
        close all;
    end    
%end