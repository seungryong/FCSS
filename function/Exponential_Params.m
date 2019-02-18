classdef Exponential_Params < dagnn.ElementWise

  methods
    function outputs = forward(obj, inputs, params)    
        outputs{1} = exp(-inputs{1}/params{1});
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)        
      derInputs{1} = derOutputs{1}.*(-1./params{1}).*exp(-inputs{1}/params{1});
      derParams_tmp = derOutputs{1}.*(inputs{1}./(params{1}*params{1})).*exp(-inputs{1}/params{1});  
      derParams{1} = sum(derParams_tmp(:));
    end

    function obj = Exponential_Params(varargin)
      obj.load(varargin) ;
    end
  end
  
end
