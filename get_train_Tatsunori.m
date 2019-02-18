clc; clear; close all;
addpath('flow-code-matlab');
%% settings
BenchmarkPath = 'datasets\TSS_CVPR2016';
if ~exist(BenchmarkPath)
    fprintf('Downloading Tatsunori benchmark...\n') ;
    mkdir('datasets','TSS_CVPR2016');
    urlwrite('http://www.hci.iis.u-tokyo.ac.jp/datasets/data/JointCorrCoseg/TSS_CVPR2016.zip', ...
    fullfile('datasets','TSS_CVPR2016','TSS_CVPR2016.zip')) ;
    fprintf('Unzipping...\n') ;
    unzip(fullfile('datasets','TSS_CVPR2016','TSS_CVPR2016.zip'),fullfile('datasets','TSS_CVPR2016'));    
end
folder_name = fullfile(BenchmarkPath,'TSS_CVPR2016');
folder_data = dir(folder_name);
folder_data = folder_data(3:end,:);

%% initialization
count_data = 0;
count_label = 0;
count_label1 = 0;

for ss = 1:length(folder_data)-1
	fprintf('image sequence: %d\n',ss);
	subfolder_name = fullfile(folder_name,folder_data(ss).name);
    subfolder_data = dir(subfolder_name);
    subfolder_data = subfolder_data(3:end,:);
    for ii = 1:length(subfolder_data)
        subsubfolder_name = fullfile(subfolder_name,subfolder_data(ii).name);    
        folder_name_image1 = fullfile(subsubfolder_name,'image1.png');
        folder_name_image2 = fullfile(subsubfolder_name,'image2.png');
        folder_name_mask1 = fullfile(subsubfolder_name,'mask1.png');
        folder_name_mask2 = fullfile(subsubfolder_name,'mask2.png');

        count_data = count_data + 1; data{count_data} = folder_name_image1;
        count_data = count_data + 1; data{count_data} = folder_name_image2;
        count_label = count_label + 1; label{count_label} = folder_name_mask1;
        count_label = count_label + 1; label{count_label} = folder_name_mask2;
    end
end

imdb = struct;
imdb.images.data = data;
imdb.images.label = label;
imdb.images.set = single(ones(1,length(data)));
save('data/imdb_correspondence_Tatsunori.mat','imdb','-v7.3');
