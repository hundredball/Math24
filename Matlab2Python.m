clear all; clc; close all;

% Select data
T_data = readtable('./Data_Matlab/data_list.xlsx');
data_list = table2array(T_data);

% % Load default channel coordinates
% default_loc = readlocs('./Channel_coordinate/Standard-10-20-Cap81.locs');

% Load channel of interest
% T = readtable('./Channel_coordinate/Channel_location_angle.xlsx');
% labels_interest = table2array(T(1:end, 1));  % labels of interest

for i_data = 1:size(data_list, 1)
    % Load data
    fileName = data_list{i_data};
    EEG = pop_loadset(['./Data_Matlab/' fileName '.set']);

%     % Get coordinate of channel in EEG
%     radius_EEG = cell2mat({EEG.chanlocs.radius});
%     theta_EEG = mod(cell2mat({EEG.chanlocs.theta})+360, 360)/360;
% 
%     % Find matched channels by sph_theta, sph_phi
%     for i = 1:size(labels_interest, 1)
%         label = labels_interest{i};
%         indice_in_default = find(ismember({default_loc.labels}, {label}));
%         indice_in_EEG = find(ismember({EEG.chanlocs.labels}, {label}));
% 
%         % Remove repetitive label first
%         if ~isempty(indice_in_EEG)
%             %fprintf(['[X]' EEG.chanlocs(indice_in_EEG).labels '->XX\n']);
%             EEG.chanlocs(indice_in_EEG).labels = 'XX';
%         end
% 
%         if ~isempty(indice_in_default)
%             radius = default_loc(indice_in_default).radius;
%             theta = mod(default_loc(indice_in_default).theta+360, 360)/360;
% 
%             % Choose channel with the smallest difference of phi and theta
%             [value, index] = min( abs(radius_EEG-radius)+abs(theta_EEG-theta));
%             radius_EEG(index) = nan;
%             theta_EEG(index) = nan;
% 
% 
%             %fprintf(['[O]' EEG.chanlocs(index).labels '->' label '\n']);
%             EEG.chanlocs(index).labels = label;
%         end
%     end
% 
%     % Reserve channels of interest
%     EEG = pop_select(EEG, 'channel', labels_interest);

    % Save channel data, events and location names
    data = EEG.data;
    event = {EEG.event.latency;EEG.event.type}';
    chanlocs = EEG.chanlocs;
    %save(['./Data_Python/' fileName(1:6) '.mat'], 'data', 'event', 'chanlocs');
    fprintf([int2str(i_data) '. Save ' fileName(1:6) '.mat\n'])
end