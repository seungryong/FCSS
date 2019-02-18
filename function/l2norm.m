classdef l2norm < dagnn.ElementWise

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnnormalizelp(inputs{1}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnnormalizelp(inputs{1}, derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = l2norm(varargin)
      obj.load(varargin) ;
    end
  end
  
end
