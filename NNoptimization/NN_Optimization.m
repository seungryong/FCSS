function [flow,warp1,warp2] = NN_Optimization(feature1, feature2, im1color, im2color, mask1, mask2, stride)

    [XX1, YY1] = meshgrid(stride:stride:size(mask1, 2), stride:stride:size(mask1, 1));
    [XX2, YY2] = meshgrid(stride:stride:size(mask2, 2), stride:stride:size(mask2, 1));
    
    sample_inds1 = sub2ind(size(mask1), YY1(:), XX1(:));
    keep1 = (mask1(sample_inds1) == 1);
    pts1 = [XX1(keep1), YY1(keep1)];
    sample_inds_color = sub2ind(size(mask1), YY1(keep1), XX1(keep1));
    
    sample_inds2 = sub2ind(size(mask2), YY2(:), XX2(:));
    keep2 = (mask2(sample_inds2) == 1);
    pts2 = [XX2(keep2), YY2(keep2)];

	f1 = zeros(length(pts1),size(feature1,3),'single');
	f2 = zeros(length(pts2),size(feature1,3),'single');
    
	for ii = 1:length(pts1)
        f1(ii,:) = feature1(pts1(ii,2),pts1(ii,1),:);
    end    
	for ii = 1:length(pts2)
        f2(ii,:) = feature2(pts2(ii,2),pts2(ii,1),:);
	end

    [match_index1,~] = vl_kdtreequery(vl_kdtreebuild(f2'), f2', f1');
    
    flow = zeros(size(mask1,1),size(mask1,2),2,'single');
    for ii = 1:length(pts1)
        flow(pts1(ii,2),pts1(ii,1),1) = pts2(match_index1(ii),2) - pts1(ii,2);
        flow(pts1(ii,2),pts1(ii,1),2) = pts2(match_index1(ii),1) - pts1(ii,1);
    end

	color_jet = colormap(jet);
    sample_inds_color = mat2gray(sample_inds_color);
    color_circle = color_jet(floor(sample_inds_color*63)+1,:);
    
	h = figure; imshow(uint8(im1color))
	for ii = 1:length(pts1)
        hold on; plot(pts1(ii,1), pts1(ii,2),'o','MarkerFaceColor',color_circle(ii,:),'MarkerEdgeColor',color_circle(ii,:));
    end
	implot = getframe;
    warp1 = implot.cdata;
	close(h);
    
    h = figure; imshow(uint8(im2color))
	for ii = 1:length(pts1)
        hold on; plot(pts2(match_index1(ii),1), pts2(match_index1(ii),2),'o','MarkerFaceColor',color_circle(ii,:),'MarkerEdgeColor',color_circle(ii,:));
    end
	implot = getframe;
    warp2 = implot.cdata;
	close(h);    
    
end