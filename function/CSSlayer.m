classdef CSSlayer < dagnn.ElementWise
    
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
      [H,W,~,nbatch] = size(inputs{1});
      
      rp1 = params{1};
      rp2 = params{2};
      rp3 = params{3};
      rp4 = params{4};
      
      obj.Ho = H;
      obj.Wo = W;      
      obj.initGrid();
      
      %%   Shifting-Transformer 1
      xxyy_sx = repmat(obj.xx,1,nbatch)+(2/obj.Ho)*rp1; 
      xxyy_sy = repmat(obj.yy,1,nbatch)+(2/obj.Wo)*rp2; 
      xxyy_sx = reshape(xxyy_sx,obj.Ho,obj.Wo,1,nbatch);
      xxyy_sy = reshape(xxyy_sy,obj.Ho,obj.Wo,1,nbatch);
      xxyy_s = cat(3,xxyy_sx,xxyy_sy);
      xxyy_s = permute(xxyy_s, [3,1,2,4]);
      sft1 = vl_nnbilinearsampler(inputs{1},xxyy_s);
      
      %%   Shifting-Transformer 2
      xxyy_tx = repmat(obj.xx,1,nbatch)+(2/obj.Ho)*rp3; 
      xxyy_ty = repmat(obj.yy,1,nbatch)+(2/obj.Wo)*rp4; 
      xxyy_tx = reshape(xxyy_tx,obj.Ho,obj.Wo,1,nbatch);
      xxyy_ty = reshape(xxyy_ty,obj.Ho,obj.Wo,1,nbatch);
      xxyy_t = cat(3,xxyy_tx,xxyy_ty);
      xxyy_t = permute(xxyy_t, [3,1,2,4]);
      sft2 = vl_nnbilinearsampler(inputs{1},xxyy_t);    
      
      %% Euclidean Dist.
      outputs{1} = sum((sft1-sft2).^2,3);  
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)            
        [~,~,depth,nbatch] = size(inputs{1});
        
        rp1 = params{1};
        rp2 = params{2};
        rp3 = params{3};
        rp4 = params{4};
               
        %% Forward-Propropagation    
        %%   Shifting-Transformer 1
        xxyy_sx = repmat(obj.xx,1,nbatch)+(2/obj.Ho)*rp1; 
        xxyy_sy = repmat(obj.yy,1,nbatch)+(2/obj.Wo)*rp2; 
        xxyy_sx = reshape(xxyy_sx,obj.Ho,obj.Wo,1,nbatch);
        xxyy_sy = reshape(xxyy_sy,obj.Ho,obj.Wo,1,nbatch);
        xxyy_s = cat(3,xxyy_sx,xxyy_sy);
        xxyy_s = permute(xxyy_s, [3,1,2,4]);
        sft1 = vl_nnbilinearsampler(inputs{1},xxyy_s);
      
        %%   Shifting-Transformer 2
        xxyy_tx = repmat(obj.xx,1,nbatch)+(2/obj.Ho)*rp3; 
        xxyy_ty = repmat(obj.yy,1,nbatch)+(2/obj.Wo)*rp4; 
        xxyy_tx = reshape(xxyy_tx,obj.Ho,obj.Wo,1,nbatch);
        xxyy_ty = reshape(xxyy_ty,obj.Ho,obj.Wo,1,nbatch);
        xxyy_t = cat(3,xxyy_tx,xxyy_ty);
        xxyy_t = permute(xxyy_t, [3,1,2,4]);
        sft2 = vl_nnbilinearsampler(inputs{1},xxyy_t);   
        
        %% Backward-Propropagation          
        derOutputs_s =  2*repmat(derOutputs{1},1,1,depth,1).*(sft1-sft2);
        derOutputs_t =  2*repmat(derOutputs{1},1,1,depth,1).*(sft2-sft1);   
        
        [inv_sft1,dG1] = vl_nnbilinearsampler(inputs{1}, xxyy_s, derOutputs_s);
        [inv_sft2,dG2] = vl_nnbilinearsampler(inputs{1}, xxyy_t, derOutputs_t);
        
        %% derInputs
        derInputs{1} = inv_sft1+inv_sft2;
        
        %% derParams
        derParams{1} = sum(dG1(1,:));         
        derParams{2} = sum(dG1(2,:));         
        derParams{3} = sum(dG2(1,:));            
        derParams{4} = sum(dG2(2,:));   
    end

    function obj = CSSlayer(varargin)
      obj.load(varargin);
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
