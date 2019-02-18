function [warpI2,flow,flow_overlay] = SIFTflow_Optimization(feature1, feature2, im1color, im2color,scale_factor)
    %% SIFT Flow Estimation and Warping
    SIFTflowpara.alpha=2*scale_factor;
    SIFTflowpara.d=40*scale_factor;
    SIFTflowpara.gamma=0.005*scale_factor;
    SIFTflowpara.nlevels=4;
    SIFTflowpara.wsize=2;
    SIFTflowpara.topwsize=10;
    SIFTflowpara.nIterations=60;
    [vx,vy,~]=SIFTflowc2f(feature1,feature2,SIFTflowpara);

    %% Flow Visualization and Warping 
    warpI2 = warpImage(single(im2color),vx,vy);
    
    clear flow;
    flow(:,:,1)=vx;
    flow(:,:,2)=vy;
    
    flow_overlay = createOverlayImage(im1color,flowToColor(flow));
       
end