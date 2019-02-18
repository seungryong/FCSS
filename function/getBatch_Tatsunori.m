function inputs = getBatch_Tatsunori(imdb,batch)
    H = 300; W = 400;
    
    img1 = gpuArray(zeros(H,W,3,'single'));
    img2 = gpuArray(zeros(H,W,3,'single'));
    img1_mask = gpuArray(zeros(H,W,'single'));
    img2_mask = gpuArray(zeros(H,W,'single'));

    image1 = gpuArray(single(imread(imdb.images.data{batch(1)})));
    image2 = gpuArray(single(imread(imdb.images.data{batch(2)})));
    image1_mask = single(imread(imdb.images.label{batch(1)}));
    image2_mask = single(imread(imdb.images.label{batch(2)}));

    img1_col_ratio = size(image1,2)/W;
    if size(image1,1)/img1_col_ratio >= H
        img1_row_ratio = size(image1,1)/H;
        img1(:,1:ceil(size(image1,2)/img1_row_ratio),:) = imresize(image1,[H,ceil(size(image1,2)/img1_row_ratio)]);
        img1_mask(:,1:ceil(size(image1,2)/img1_row_ratio),:) = gpuArray(imresize(image1_mask,[H,ceil(size(image1,2)/img1_row_ratio)],'neares'));
    else
        img1(1:ceil(size(image1,1)/img1_col_ratio),:,:) = imresize(image1,[ceil(size(image1,1)/img1_col_ratio),W]);
        img1_mask(1:ceil(size(image1,1)/img1_col_ratio),:,:) = gpuArray(imresize(image1_mask,[ceil(size(image1,1)/img1_col_ratio),W],'neares'));
    end
    img2_col_ratio = size(image2,2)/W;
    if size(image2,1)/img2_col_ratio >= H
        img2_row_ratio = size(image2,1)/H;
        img2(:,1:ceil(size(image2,2)/img2_row_ratio),:) = imresize(image2,[H,ceil(size(image2,2)/img2_row_ratio)]);
        img2_mask(:,1:ceil(size(image2,2)/img2_row_ratio),:) = imresize(image2_mask,[H,ceil(size(image2,2)/img2_row_ratio)],'neares');
    else
        img2(1:ceil(size(image2,1)/img2_col_ratio),:,:) = imresize(image2,[ceil(size(image2,1)/img2_col_ratio),W]);
        img2_mask(1:ceil(size(image2,1)/img2_col_ratio),:,:) = imresize(image2_mask,[ceil(size(image2,1)/img2_col_ratio),W],'neares');
    end

    img1_mask(img1_mask>1) = 1;
    img2_mask(img2_mask>1) = 1;
    
    data_temp(:, :, :, 1) = gpuArray(single(img1));
    data_temp(:, :, :, 2) = gpuArray(single(img2));

    label_temp(:, :, 1, 1) = gpuArray(single(img1_mask));
    label_temp(:, :, 1, 2) = gpuArray(single(img2_mask)); 
    
    label_temp(:, :, 2:4, 1) = gpuArray(single(img1));
    label_temp(:, :, 2:4, 2) = gpuArray(single(img2));
    
    inputs = {'input', data_temp, 'label', label_temp};
end