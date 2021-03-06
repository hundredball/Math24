clear all; clc; close all;

% Select data
T_data = readtable('../Data_Matlab/data_list.xlsx');
data_list = table2array(T_data);

% % Load default channel coordinates
% default_loc = readlocs('./Channel_coordinate/Standard-10-20-Cap81.locs');

% Load channel of interest
T = readtable('../Channel_coordinate/Channel_location_angle_21.xlsx');
labels_interest = table2array(T(1:end, 1));  % labels of interest
coord_interest = table2array(T(1:end, 2:3)); % coordinate of interest

for i_data = 1:size(data_list, 1)
    % Load data
    fileName = data_list{i_data};
    EEG = pop_loadset(['../Data_Matlab/' fileName '.set']);

    % Get coordinate of channel in EEG
    radius_EEG = cell2mat({EEG.chanlocs.radius});
    theta_EEG = cell2mat({EEG.chanlocs.theta});

    % Find nearest channels
    for i = 1:size(labels_interest, 1)
        label = labels_interest{i};
        indice_in_EEG = find(ismember({EEG.chanlocs.labels}, {label}));

        % Remove repetitive label first
        if ~isempty(indice_in_EEG)
            fprintf(['[X]' EEG.chanlocs(indice_in_EEG).labels '->XX\n']);
            EEG.chanlocs(indice_in_EEG).labels = 'XX';
        end

        radius = coord_interest(i,1);
        theta = coord_interest(i,2);

        % Choose channel with shortest distance
        [value, index] = min( radius_EEG.^2+radius^2 - 2*radius*radius_EEG.*cos((theta_EEG-theta)*pi/180) );
        radius_EEG(index) = nan;
        theta_EEG(index) = nan;

        % fprintf(['[O]' EEG.chanlocs(index).labels '->' label '\n']);
        EEG.chanlocs(index).labels = label;
    end

    % Reserve channels of interest
    EEG = pop_select(EEG, 'channel', labels_interest);

    % Save channel data, events and location names
    data = EEG.data;
    event = {EEG.event.latency;EEG.event.type}';
    chanlocs = EEG.chanlocs;
    save(['../Data_Python/' fileName(1:6) '_21.mat'], 'data', 'event', 'chanlocs');
    fprintf([int2str(i_data) '. Save ' fileName(1:6) '_21.mat\n'])
end

% Plot channel locations
% topoplot([],EEG.chanlocs,'style','blank','electrodes','labelpoint','chaninfo',EEG.chaninfo);
