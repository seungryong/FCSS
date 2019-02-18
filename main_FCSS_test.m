%% Script for testing FCSS descriptor network
%% Written by Seungryong Kim, Yonsei University, Seoul, Korea
clc; clear; close all;
run('vlfeat-0.9.20\toolbox\vl_setup');
run('matconvnet-1.0-beta23\matlab\vl_setupnn.m');
addpath('SIFTflow');
addpath('PFflow');
addpath('NNoptimization');
addpath('flow-code-matlab');
addpath('init_model');
addpath('model');
addpath('function');

folder_name = 'images';
folder_data = dir(folder_name);
folder_data = folder_data(3:end,:);

save_experiments_folder = 'results'; 

for ii = 1:length(folder_data)
    fprintf('image sequence: %d\n',ii);
    
	subfolder_name = fullfile(folder_name,folder_data(ii).name);
    im1color = double(imread(fullfile(subfolder_name,'image1.png')));
    im2color = double(imread(fullfile(subfolder_name,'image2.png')));
    mask1 = double(imread(fullfile(subfolder_name,'mask1.png')));
    mask2 = double(imread(fullfile(subfolder_name,'mask2.png')));
    mask1(mask1==255) = 1;
    mask2(mask2==255) = 1;    
    
    im1color = imresize(im1color,0.75);
    im2color = imresize(im2color,0.75);    
    mask1 = imresize(mask1,0.75,'nearest');
    mask2 = imresize(mask2,0.75,'nearest');
    
    mkdir(fullfile(save_experiments_folder,folder_data(ii).name));
    
    warpI2_SF_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'warpI2_SF.png');
    flow_SF_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'flow_SF.png');
    flow_overlay_SF_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'flow_overlay_SF.png');
    
    warpI2_PF_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'warpI2_PF.png');
    flow_PF_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'flow_PF.png');
    flow_overlay_PF_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'flow_overlay_PF.png');
    show_match_PF_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'show_match_PF.png');    
    
    visualization_NN_img1_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'warpI1_NN.png');
    visualization_NN_img2_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'warpI2_NN.png');
    flow_overlay_NN_dir = fullfile(fullfile(save_experiments_folder,folder_data(ii).name),'flow_overlay_NN.png');
    
    %% Image Size Check
    [row1,col1,~] = size(im1color);
    if mod(row1,4) ~= 0
        im1color = im1color(1:row1-mod(row1,4),:,:);
        mask1 = mask1(1:row1-mod(row1,4),:,:);
    end
    if mod(col1,4) ~= 0
        im1color = im1color(:,1:col1-mod(col1,4),:,:);
        mask1 = mask1(:,1:col1-mod(col1,4),:,:);
    end
    [row1,col1,~] = size(im1color);

    [row2,col2,~] = size(im2color);
    if mod(row2,4) ~= 0
        im2color = im2color(1:row2-mod(row2,4),:,:);
        mask2 = mask2(1:row2-mod(row2,4),:,:);
    end
    if mod(col2,4) ~= 0
        im2color = im2color(:,1:col2-mod(col2,4),:,:);
        mask2 = mask2(:,1:col2-mod(col2,4),:,:);
    end
    [row2,col2,~] = size(im2color);
    figure(1); imshow(uint8(im1color)); title('Image 1');
    figure(2); imshow(uint8(im2color)); title('Image 2');
    
    %% FCSS Descriptor (Pre-trained Model)
    load('data/fcss/net-epoch.mat');

    net_FCSS = dagnn.DagNN.loadobj(net);
    net_FCSS.mode = 'test';
    net_FCSS.move('gpu');
    net_FCSS_ind = net_FCSS.getVarIndex('FCSS_feature');
    net_FCSS.vars(net_FCSS_ind).precious = 1;

    inputs1 = {'input', gpuArray(single(im1color))};
    net_FCSS.eval(inputs1);
    dense_cnn1_fcss = gather(net_FCSS.vars(net_FCSS_ind).value);
    inputs2 = {'input', gpuArray(single(im2color))};
    net_FCSS.eval(inputs2);
    dense_cnn2_fcss = gather(net_FCSS.vars(net_FCSS_ind).value);

    %% SIFT Flow Estimation and Warping
    scale_factor = 1.0;
    [warpI2,flow,flow_overlay] = SIFTflow_Optimization(dense_cnn1_fcss,dense_cnn2_fcss,uint8(gather(im1color)),uint8(gather(im2color)),scale_factor);
    figure(3); imshow(uint8(warpI2)); title('Warped Image 2 with SF flow');
    figure(4); imshow(flowToColor(flow)); title('Flow Visualization with SF flow');
    figure(5); imshow(flow_overlay); title('Flow Visualization with SF flow');
    imwrite(uint8(warpI2), warpI2_SF_dir); 
    imwrite(uint8(flowToColor(flow)), flow_SF_dir); 
    imwrite(uint8(flow_overlay), flow_overlay_SF_dir); 

    %% PF Flow Estimation and Warping
    [warpI2,flow,flow_overlay,ShowMatch] = PFflow_Optimization(uint8(gather(im1color)),uint8(gather(im2color)));
    figure(6); imshow(uint8(warpI2)); title('Warped Image 2 with PF flow');
    figure(7); imshow(flowToColor(flow)); title('Flow Visualization with PF flow');
    figure(8); imshow(flow_overlay); title('Flow Visualization with PF flow');
    figure(9); imshow(ShowMatch); title('Show Matches with PF flow');
    imwrite(uint8(warpI2), warpI2_PF_dir); 
    imwrite(uint8(flowToColor(flow)), flow_PF_dir); 
    imwrite(uint8(flow_overlay), flow_overlay_PF_dir); 
    imwrite(uint8(ShowMatch), show_match_PF_dir); 

    %% Nearest-Neighbor (NN) optimization
    stride = 8;
    [flow,warp1,warp2] = NN_Optimization(dense_cnn1_fcss,dense_cnn2_fcss,uint8(im1color),uint8(im2color),mask1,mask2,stride);
    figure(10); imshow(flowToColor(flow)); title('Flow Visualization with NN');
    figure(11); imshow(warp1); title('NN Visualization of Image 1');
    figure(12); imshow(warp2); title('NN Visualization of Image 2');
    imwrite(uint8(flowToColor(flow)), visualization_NN_img1_dir); 
    imwrite(uint8(warp1), visualization_NN_img2_dir); 
    imwrite(uint8(warp2), flow_overlay_NN_dir);    
end