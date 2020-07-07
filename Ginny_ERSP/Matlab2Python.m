clear all; clc; close all;

% Select data
root_path = matlab.desktop.editor.getActiveFilename;
root_path = root_path(1:end-26);
T_data = readtable([root_path 'Data_Matlab/data_list.xlsx']);
data_list = table2array(T_data);

num_channel = 12;
freq_step = 114;
time_step = 200;

for i_data = 1:size(data_list, 1)
    
    % Set folder path of the file
    date = data_list{i_data}(1:6);
    date_path = [root_path 'savedata/' date '/'];
    
    % Load baseline frequency
    load([date_path date '_diff2-baselinefreqs.mat'], 'freqs');
    
    % Load three kinds of time (baseline, cue, SL)
    load([date_path date '_diff2-tmpfordiff2.mat'], 'tmp');
    num_epoch = size(tmp, 1);
    
    for i_epoch = 1:num_epoch
        for i_channel = 1:num_channel
            fileName = [date_path date '_diff2channel' num2str(i_channel) 'epoch' ...
                num2str(i_epoch) '-baselineersp.mat'];
            if i_epoch==1 && i_channel==1
                ERSP = zeros(num_epoch, num_channel, freq_step, time_step);
            end
            load(fileName, 'ersp');
            ERSP(i_epoch, i_channel, :, :) = ersp;
        end
    end
    
    % Save channel data, events and location names
    fileName = [date_path date '_python.mat'];
    save(fileName, 'ERSP', 'tmp', 'freqs');
    fprintf([int2str(i_data) '. Save ' date '_python.mat\n'])
end