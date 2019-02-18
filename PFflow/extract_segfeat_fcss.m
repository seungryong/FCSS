function [ feat ] = extract_segfeat_fcss(img, seg)
    %% extract fcss features from segments

    %% FCSS Descriptor (Pre-trained Model)
    load('data/fcss/net-epoch.mat');

    net_FCSS = dagnn.DagNN.loadobj(net);
    net_FCSS.mode = 'test';
    net_FCSS.move('gpu');
    net_FCSS_ind = net_FCSS.getVarIndex('FCSS_feature');
    net_FCSS.vars(net_FCSS_ind).precious = 1;

    % initialize structs
    feat = struct;

    imsize_cnn = [96,96]; 

    heights = double(seg.coords(:,3) - seg.coords(:,1) + 1);
    widths = double(seg.coords(:,4) - seg.coords(:,2) + 1);
    box_rects = [ seg.coords(:,1:2) heights widths ];
    
    ims_ = zeros([imsize_cnn(1), imsize_cnn(2), 3, size(seg.coords,1)], 'single');

    for j = 1:size(seg.coords,1)
        img_patch_o = imresize(imcrop(img, box_rects(j,:)), imsize_cnn);
        ims_(:,:,:,j) = single(img_patch_o);
    end
    
	inputs1 = {'input', gpuArray(ims_)};
	net_FCSS.eval(inputs1);        
	dense_cnn_fcss = gather(net_FCSS.vars(net_FCSS_ind).value);
	dense_cnn_fcss = dense_cnn_fcss(48,48,:,:);    
    cnn_temp = reshape(dense_cnn_fcss, numel(dense_cnn_fcss)/size(seg.coords,1), size(seg.coords,1))';

    % add to feat
    feat.hist = single(cnn_temp);
    feat.boxes = seg.coords;
    feat.img = img; 

end
