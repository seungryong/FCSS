classdef CSSlayer_shift < dagnn.ElementWise
    
    properties
        Ho = 0;
        Wo = 0;
    end
    
	properties (Transient)
        xx;
        yy;
	end

  methods
    function outputs = forward(obj, inputs, params)     
        rp1 = round(params{1});
        rp2 = round(params{2});
        rp3 = round(params{3});
        rp4 = round(params{4});
      
        %%   Shifting-Transformer 1
        sft1 = shift_act(inputs{1},[rp1,rp2]);     
        
        %%   Shifting-Transformer 2
        sft2 = shift_act(inputs{1},[rp3,rp4]);        
        
        %% Euclidean Dist.
        outputs{1} = sum((sft1-sft2).^2,3);  
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)            
        [~,~,depth,~] = size(inputs{1});
        
        rp1 = round(params{1});
        rp2 = round(params{2});
        rp3 = round(params{3});
        rp4 = round(params{4});
               
        %% Forward-Propropagation   
        %%   Shifting-Transformer 1
        sft1 = shift_act(inputs{1},[rp1,rp2]);      
        
        %%   Shifting-Transformer 2
        sft2 = shift_act(inputs{1},[rp3,rp4]);   
        
        [derInputs_ws_dx,derInputs_ws_dy] = gradient(sft1); 
        [derInputs_wt_dx,derInputs_wt_dy] = gradient(sft2); 
        
        %% Backward-Propropagation          
        derOutputs_s =  2*repmat(derOutputs{1},1,1,depth,1).*(sft1-sft2);
        derOutputs_t =  2*repmat(derOutputs{1},1,1,depth,1).*(sft2-sft1);   
        
        %% Backward Shifting-Transformer 1
        inv_sft1 = shift_act(derOutputs_s,-[rp1,rp2]);      
        
        %% Backward Shifting-Transformer 2
        inv_sft2 = shift_act(derOutputs_t,-[rp3,rp4]);          
        
        %% derInputs
        derInputs{1} = inv_sft1+inv_sft2;             
        %% derParams
        derParams{1} = sum(derOutputs_s(:).*derInputs_ws_dx(:));
        derParams{2} = sum(derOutputs_s(:).*derInputs_ws_dy(:)); 
        derParams{3} = sum(derOutputs_t(:).*derInputs_wt_dx(:)); 
        derParams{4} = sum(derOutputs_t(:).*derInputs_wt_dy(:)); 
    end

    function obj = CSSlayer_shift(varargin)
      obj.load(varargin) ;
    end
    
    function initGrid(obj)
        xi = linspace(-1, 1, obj.Ho);
        yi = linspace(-1, 1, obj.Wo);
        [y,x] = meshgrid(yi, xi);
        x = gpuArray(x(:));
        y = gpuArray(y(:));
        obj.xx = x;
        obj.yy = y;
    end
    
  end
  
end
