classdef CorrespondenceLoss < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)    
        num_train = 64;
        stride = 3;
        threshold = 6;
%         threshold_C = 0.9;
                
        [~,~,feat_length,~] = size(inputs{1});
        
        mask1 = gather(inputs{2}(:,:,1,1));        
        mask2 = gather(inputs{2}(:,:,1,2));   
        
        features1 = gather(inputs{1}(:,:,:,1));
        features2 = gather(inputs{1}(:,:,:,2));
        
        image1 = uint8(gather(inputs{2}(:,:,2:4,1)));
        image2 = uint8(gather(inputs{2}(:,:,2:4,2)));
        
        %% Candidate Sampling
        [points1y, points1x] = meshgrid(stride:stride:size(mask1, 2), stride:stride:size(mask1, 1));    
        sample_inds1 = sub2ind(size(mask1), points1x(:), points1y(:));
        keep1 = (mask1(sample_inds1) == 1);
        points1x = points1x(keep1); 
        points1y = points1y(keep1); 
        
        [points2y, points2x] = meshgrid(stride:stride:size(mask2, 2), stride:stride:size(mask2, 1));    
        sample_inds2 = sub2ind(size(mask2), points2x(:), points2y(:));
        keep2 = (mask2(sample_inds2) == 1);
        points2x = points2x(keep2); 
        points2y = points2y(keep2); 
        
        features1_samples = zeros(length(points1x),feat_length,'single');
        for ii = 1:length(points1x), features1_samples(ii,:) = features1(points1x(ii),points1y(ii),:); end               
        features2_samples = zeros(length(points2x),feat_length,'single');
        for ii = 1:length(points2x), features2_samples(ii,:) = features2(points2x(ii),points2y(ii),:); end   
        
        %% K-NN Search
        [matched_id1,matched_dist1] = vl_kdtreequery(vl_kdtreebuild(features2_samples'), features2_samples', features1_samples');
        [matched_id2,~] = vl_kdtreequery(vl_kdtreebuild(features1_samples'), features1_samples', features2_samples');

        count = 0; %count1 = 0;
        for ii = 1:length(points1x)
            if abs(points1x(ii)-points1x(matched_id2(matched_id1(ii)))) + abs(points1y(ii)-points1y(matched_id2(matched_id1(ii)))) <= threshold
                count = count + 1;
                index1_match(count,1) = ii;
                index2_match(count,1) = matched_id1(ii);
                cost_match(count,1) = matched_dist1(ii);
%             else
%                 count1 = count1 + 1;
%                 index1_unmatch(count1,1) = ii;
%                 index2_unmatch(count1,1) = matched_id1(ii);       
%                 cost_unmatch(count1,1) = matched_dist1(ii);
            end
        end        
        
        if count < num_train, num_train = count; end 
        
        % Positive Sampling
        [~, index_hard_positive] = sort(cost_match,'ascend');   
%          index_hard_positive = randperm(length(cost_match))';
        index1_match = index1_match(index_hard_positive);
        index2_match = index2_match(index_hard_positive);
        cost_match = cost_match(index_hard_positive);
        index1_match = index1_match(1:num_train);
        index2_match = index2_match(1:num_train);
        cost_match = cost_match(1:num_train);
        
        %% Matching Visualization (Positive)
        matchedPoints1 = [points1y(index1_match(:)),points1x(index1_match(:))];
        matchedPoints2 = [points2y(index2_match(:)),points2x(index2_match(:))];
        figure(2); showMatchedFeatures(image1,image2,matchedPoints1,matchedPoints2,'montage'); title('Positive Samples');
%         color_jet = colormap(jet);
%         sample_inds_color = mat2gray(sub2ind(size(mask1),points1x(index1_match(:)),points1y(index1_match(:))));
%         color_circle = color_jet(floor(sample_inds_color*63)+1,:);
% 
%         figure; imshow([image1,image2]); title('Positive Samples');
%         for ii = 1:length(index1_match)
%             hold on; plot(points1y(index1_match(ii)),points1x(index1_match(ii)),'o','MarkerFaceColor',color_circle(ii,:),'MarkerEdgeColor',color_circle(ii,:),'MarkerSize',3);
%             hold on; plot(size(image1,2)+points2y(index2_match(ii)),points2x(index2_match(ii)),'o','MarkerFaceColor',color_circle(ii,:),'MarkerEdgeColor',color_circle(ii,:),'MarkerSize',3);
%         end
                
        %% Loss Function
        %%%% Correspondence Contrastive Loss %%%%
        % Negative Sampling
%         num_train_neg = num_train*3;
%         if count1 < num_train_neg, num_train_neg = count1; end
% 
% %         [~, index_hard_negative] = sort(cost_unmatch,'ascend');   
%         index_hard_negative = randperm(length(cost_unmatch))';
%         index1_unmatch = index1_unmatch(index_hard_negative);
%         index2_unmatch = index2_unmatch(index_hard_negative);
%         index1_unmatch = index1_unmatch(1:num_train_neg);
%         index2_unmatch = index2_unmatch(1:num_train_neg);        
% 
%         features1_matched = features1_samples(index1_match,:);
%         features2_matched = features2_samples(index2_match,:);        
%         positive_sum = sum(sum((features1_matched-features2_matched).^2,2),1);
%         
%         features1_unmatched = features1_samples(index1_unmatch,:);
%         features2_unmatched = features2_samples(index2_unmatch,:);        
%         negative_diff = repmat(threshold_C,num_train_neg,1) - sum((features1_unmatched-features2_unmatched).^2,2);
%         negative_diff_idx = (negative_diff > 0);
%         negative_sum = sum(negative_diff.*negative_diff_idx,1);
%         
%         outputs{1} = sum(positive_sum+negative_sum)./(num_train+num_train_neg);   
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%% Correspondence Classification Loss %%%%
        num_train_neg = num_train*3;
        index2_unmatch = matched_id1(randperm(length(matched_id1))');
        index2_unmatch = index2_unmatch(1:num_train_neg); 
        
        features1_matched = features1_samples(index1_match,:);
        features2_unmatched = features2_samples(index2_unmatch,:);
        
        norm_matrix = vl_alldist2(features1_matched',features2_unmatched','L2'); 
        norm_matrix = [cost_match,norm_matrix];
        norm_matrix = 1 - norm_matrix;       
        
        Xmax = max(norm_matrix,[],2);        
        ex = exp(norm_matrix - repmat(Xmax,1,1+num_train_neg));
        t = Xmax + log(sum(ex,2)) - norm_matrix(:,1);

        outputs{1} = sum(t(:))/num_train;     
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                         
        n = obj.numAveraged;
        m = n + size(inputs{1},4);
        obj.average = (n * obj.average + double(gather(outputs{1}))) / m;
        obj.numAveraged = m;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)        
        num_train = 64;
        stride = 3;
        threshold = 6;
%         threshold_C = 0.9;
                
        [H,W,feat_length,~] = size(inputs{1});
        backward1 = zeros(H,W,feat_length,'single');
        backward2 = zeros(H,W,feat_length,'single');
        
        mask1 = gather(inputs{2}(:,:,1,1));      
        mask2 = gather(inputs{2}(:,:,1,2));   
       
        features1 = gather(inputs{1}(:,:,:,1));
        features2 = gather(inputs{1}(:,:,:,2));

        %% Candidate Sampling
        [points1y, points1x] = meshgrid(stride:stride:size(mask1, 2), stride:stride:size(mask1, 1));    
        sample_inds1 = sub2ind(size(mask1), points1x(:), points1y(:));
        keep1 = (mask1(sample_inds1) == 1);
        points1x = points1x(keep1); 
        points1y = points1y(keep1); 
        
        [points2y, points2x] = meshgrid(stride:stride:size(mask2, 2), stride:stride:size(mask2, 1));    
        sample_inds2 = sub2ind(size(mask2), points2x(:), points2y(:));
        keep2 = (mask2(sample_inds2) == 1);
        points2x = points2x(keep2); 
        points2y = points2y(keep2); 
        
        features1_samples = zeros(length(points1x),feat_length,'single');
        for ii = 1:length(points1x), features1_samples(ii,:) = features1(points1x(ii),points1y(ii),:); end               
        features2_samples = zeros(length(points2x),feat_length,'single');
        for ii = 1:length(points2x), features2_samples(ii,:) = features2(points2x(ii),points2y(ii),:); end   
        
        %% K-NN Search
        [matched_id1,matched_dist1] = vl_kdtreequery(vl_kdtreebuild(features2_samples'), features2_samples', features1_samples');
        [matched_id2,~] = vl_kdtreequery(vl_kdtreebuild(features1_samples'), features1_samples', features2_samples');

        count = 0; %count1 = 0;
        for ii = 1:length(points1x)
            if abs(points1x(ii)-points1x(matched_id2(matched_id1(ii)))) + abs(points1y(ii)-points1y(matched_id2(matched_id1(ii)))) <= threshold
                count = count + 1;
                index1_match(count,1) = ii;
                index2_match(count,1) = matched_id1(ii);
                cost_match(count,1) = matched_dist1(ii);
%             else
%                 count1 = count1 + 1;
%                 index1_unmatch(count1,1) = ii;
%                 index2_unmatch(count1,1) = matched_id1(ii);               
%                 cost_unmatch(count1,1) = matched_dist1(ii);
            end
        end        
        
        if count < num_train, num_train = count; end 
        
        % Positive Sampling
        [~, index_hard_positive] = sort(cost_match,'ascend');   
%          index_hard_positive = randperm(length(cost_match))';
        index1_match = index1_match(index_hard_positive);
        index2_match = index2_match(index_hard_positive);
        cost_match = cost_match(index_hard_positive);
        index1_match = index1_match(1:num_train);
        index2_match = index2_match(1:num_train);
        cost_match = cost_match(1:num_train);
                
        %% Loss Function
        %%%% Correspondence Contrastive Loss %%%%  
        % Negative Sampling
%         num_train_neg = num_train*3;
%         if count1 < num_train_neg, num_train_neg = count1; end
%         
% %         [~, index_hard_negative] = sort(cost_unmatch,'descend');   
%         index_hard_negative = randperm(length(cost_unmatch))';
%         index1_unmatch = index1_unmatch(index_hard_negative);
%         index2_unmatch = index2_unmatch(index_hard_negative);
%         index1_unmatch = index1_unmatch(1:num_train_neg);
%         index2_unmatch = index2_unmatch(1:num_train_neg);    
% 
%         for ii = 1:num_train
%             pts1_x = points1x(index1_match(ii)); pts1_y = points1y(index1_match(ii));
%             pts2_x = points2x(index2_match(ii)); pts2_y = points2y(index2_match(ii));
%             
%             feature_difference = features1(pts1_x,pts1_y,:) - features2(pts2_x,pts2_y,:);
%             backward1(pts1_x,pts1_y,:) = backward1(pts1_x,pts1_y,:) + feature_difference;
%             backward2(pts2_x,pts2_y,:) = backward2(pts2_x,pts2_y,:) - feature_difference;
%         end
%          
%         features1_unmatched = features1_samples(index1_unmatch,:);
%         features2_unmatched = features2_samples(index2_unmatch,:);        
%         negative_diff = repmat(threshold_C,num_train_neg,1) - sum((features1_unmatched-features2_unmatched).^2,2);
%         negative_diff_idx = (negative_diff > 0);
%         
%         for ii = 1:num_train_neg
%             if negative_diff_idx(ii) == 1
%                 pts1_x = points1x(index1_unmatch(ii)); pts1_y = points1y(index1_unmatch(ii));
%                 pts2_x = points2x(index2_unmatch(ii)); pts2_y = points2y(index2_unmatch(ii));
%                 
%                 feature_difference = features1(pts1_x,pts1_y,:) - features2(pts2_x,pts2_y,:);
%                 backward1(pts1_x,pts1_y,:) = backward1(pts1_x,pts1_y,:) - feature_difference;
%                 backward2(pts2_x,pts2_y,:) = backward2(pts2_x,pts2_y,:) + feature_difference;
%             end
%         end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%% Correspondence Classification Loss %%%%
        num_train_neg = num_train*3;
        index2_unmatch = matched_id1(randperm(length(matched_id1))');
        index2_unmatch = index2_unmatch(1:num_train_neg); 
        
        features1_matched = features1_samples(index1_match,:);
        features2_unmatched = features2_samples(index2_unmatch,:);
       
        norm_matrix = vl_alldist2(features1_matched',features2_unmatched','L2');
        norm_matrix = [cost_match,norm_matrix];
        norm_matrix = 1 - norm_matrix;
        
        Xmax = max(norm_matrix,[],2);        
        ex = exp(norm_matrix - repmat(Xmax,1,1+num_train_neg));
        Ymat = ex ./ repmat(sum(ex,2),1,1+num_train_neg);
        Ymat(:,1) = Ymat(:,1) - ones(num_train,1,'single');

        for ii = 1:num_train
            for jj = 1:1+num_train_neg
                if jj == 1
                    pts1_x = points1x(index1_match(ii)); pts1_y = points1y(index1_match(ii));
                    pts2_x = points2x(index2_match(ii)); pts2_y = points2y(index2_match(ii));
                else
                    pts1_x = points1x(index1_match(ii)); pts1_y = points1y(index1_match(ii));
                    pts2_x = points2x(index2_unmatch(jj-1)); pts2_y = points2y(index2_unmatch(jj-1));
                end
                
                feature_difference = features2(pts2_x,pts2_y,:) - features1(pts1_x,pts1_y,:);
                backward1(pts1_x,pts1_y,:) = backward1(pts1_x,pts1_y,:) + Ymat(ii,jj)*feature_difference;
                backward2(pts2_x,pts2_y,:) = backward2(pts2_x,pts2_y,:) - Ymat(ii,jj)*feature_difference;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        derInputs{1}(:,:,:,1) = gpuArray(backward1);
        derInputs{1}(:,:,:,2) = gpuArray(backward2);
        derInputs{2} = [];
        derParams = {};
    end

    function obj = CorrespondenceLoss(varargin)
      obj.load(varargin);
    end
  end
end
