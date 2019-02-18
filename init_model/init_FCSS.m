function net = init_FCSS(init_model)

if init_model
    load('data/fcss/net-epoch.mat');
    net = dagnn.DagNN.loadobj(net);
else    
    
%% Param. Initialization
level1_length = 64;
level2_length = 64;
level3_length = 64;

%% VGG-Net Initialization
% -------------------------------------------------------------------------
%                    Load & download the source model if needed (VGG VD 16)
% -------------------------------------------------------------------------
sourceModelPath = 'model/imagenet-vgg-verydeep-19.mat';
if ~exist(sourceModelPath)
  fprintf('downloading VGG model\n') ;
  mkdir(fileparts(sourceModelPath)) ;
  urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat', sourceModelPath) ;
end
net_vgg = load(sourceModelPath);

net.layers = {};

%% Activations until Conv3-4
for ii = 1:17
    net.layers{end+1} = net_vgg.layers{ii};
end

% net = insertBnorm(net, 1);
% net = insertBnorm(net, 4);
% net = insertBnorm(net, 8);
% net = insertBnorm(net, 11);
% net = insertBnorm(net, 15);
% net = insertBnorm(net, 18);
% net = insertBnorm(net, 21);
                       
net = vl_simplenn_tidy(net);
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);

%% Level 1
%% L2-normalization
net.addLayer('normalization_level1', l2norm(), {'x8'}, {'feature_level1'}) ;

%% Two-stream Shifting Transformer
for i = 1:level1_length    
    sf = sprintf('sf_%.2d_level1', i);    
    sf_out = sprintf('sf_d%.2d_level1', i);
    
    sf_s_dx = sprintf('sf_s_dx%.2d_level1', i);
    sf_s_dy = sprintf('sf_s_dy%.2d_level1', i);
    sf_t_dx = sprintf('sf_t_dx%.2d_level1', i);
    sf_t_dy = sprintf('sf_t_dy%.2d_level1', i);
    
    net.addLayer(sf, CSSlayer(), {'feature_level1'}, {sf_out}, {sf_s_dx,sf_s_dy,sf_t_dx,sf_t_dy});
    % net.addLayer(sf, CSSlayer_shift(), {'feature_level1'}, {sf_out}, {sf_s_dx,sf_s_dy,sf_t_dx,sf_t_dy});
    
    f = net.getParamIndex(sf_s_dx);
    net.params(f).value = randn(1, 1, 'single')*9;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;
    f = net.getParamIndex(sf_s_dy);
    net.params(f).value = randn(1, 1, 'single')*9;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;
    f = net.getParamIndex(sf_t_dx);
    net.params(f).value = randn(1, 1, 'single')*9;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;
    f = net.getParamIndex(sf_t_dy);
    net.params(f).value = randn(1, 1, 'single')*9;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;    
end

%% Concat. Network
net.addLayer('concat_level1', dagnn.Concat('dim',3),...
    {'sf_d01_level1','sf_d02_level1','sf_d03_level1','sf_d04_level1','sf_d05_level1','sf_d06_level1','sf_d07_level1','sf_d08_level1','sf_d09_level1','sf_d10_level1',...
    'sf_d11_level1','sf_d12_level1','sf_d13_level1','sf_d14_level1','sf_d15_level1','sf_d16_level1','sf_d17_level1','sf_d18_level1','sf_d19_level1','sf_d20_level1',...
    'sf_d21_level1','sf_d22_level1','sf_d23_level1','sf_d24_level1','sf_d25_level1','sf_d26_level1','sf_d27_level1','sf_d28_level1','sf_d29_level1','sf_d30_level1',...
    'sf_d31_level1','sf_d32_level1','sf_d33_level1','sf_d34_level1','sf_d35_level1','sf_d36_level1','sf_d37_level1','sf_d38_level1','sf_d39_level1','sf_d40_level1',...
    'sf_d41_level1','sf_d42_level1','sf_d43_level1','sf_d44_level1','sf_d45_level1','sf_d46_level1','sf_d47_level1','sf_d48_level1','sf_d49_level1','sf_d50_level1',...
    'sf_d51_level1','sf_d52_level1','sf_d53_level1','sf_d54_level1','sf_d55_level1','sf_d56_level1','sf_d57_level1','sf_d58_level1','sf_d59_level1','sf_d60_level1',...
    'sf_d61_level1','sf_d62_level1','sf_d63_level1','sf_d64_level1'}, {'concat_out_level1'}) ;

%% Non-linear Mapping
net.addLayer('nonlinear1', Exponential_Params(), {'concat_out_level1'}, {'nonlinear_feature1'}, {'lambda1'});
f = net.getParamIndex('lambda1');
net.params(f).value = 0.5;
net.params(f).learningRate = 0;
net.params(f).weightDecay = 0.9;  

%% Max Pooling
net.addLayer('pooling_level1', ...
  dagnn.Pooling('poolSize', [2,2], 'pad', [0,1,0,1], 'stride', [1,1], 'method', 'max'), ...
  {'nonlinear_feature1'}, {'maxpool_feature1'});

%% Bilinear Upsampling
filters = single(bilinear_u(4, 1, level1_length)) ;
net.addLayer('deconv1', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 2, ...
                      'crop', 1, ...
                      'hasBias', false), ...
             'maxpool_feature1', 'feature1', 'deconv1f');

f = net.getParamIndex('deconv1f');
net.params(f).value = filters;
net.params(f).learningRate = 0;
net.params(f).weightDecay = 1;

%% L2-normalization
net.addLayer('normalization_out1', l2norm(), {'feature1'}, {'FCSS_feature1'});

%% Level 2
%% L2-normalization
net.addLayer('normalization_level2', l2norm(), {'x13'}, {'feature_level2'});

%% Two-stream Shifting Transformer
for i = 1:level2_length
    sf = sprintf('sf_%.2d_level2', i);    
    sf_out = sprintf('sf_d%.2d_level2', i);
    
    sf_s_dx = sprintf('sf_s_dx%.2d_level2', i);
    sf_s_dy = sprintf('sf_s_dy%.2d_level2', i);
    sf_t_dx = sprintf('sf_t_dx%.2d_level2', i);
    sf_t_dy = sprintf('sf_t_dy%.2d_level2', i);
    
    net.addLayer(sf, CSSlayer(), {'feature_level2'}, {sf_out}, {sf_s_dx,sf_s_dy,sf_t_dx,sf_t_dy});
%     net.addLayer(sf, CSSlayer_shift(), {'feature_level2'}, {sf_out}, {sf_s_dx,sf_s_dy,sf_t_dx,sf_t_dy});
    
    f = net.getParamIndex(sf_s_dx);
    net.params(f).value = randn(1, 1, 'single')*7;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;
    f = net.getParamIndex(sf_s_dy);
    net.params(f).value = randn(1, 1, 'single')*7;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;
    f = net.getParamIndex(sf_t_dx);
    net.params(f).value = randn(1, 1, 'single')*7;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;
    f = net.getParamIndex(sf_t_dy);
    net.params(f).value = randn(1, 1, 'single')*7;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;    
end

%% Concat. Network
net.addLayer('concat_level2', dagnn.Concat('dim',3),...
    {'sf_d01_level2','sf_d02_level2','sf_d03_level2','sf_d04_level2','sf_d05_level2','sf_d06_level2','sf_d07_level2','sf_d08_level2','sf_d09_level2','sf_d10_level2',...
    'sf_d11_level2','sf_d12_level2','sf_d13_level2','sf_d14_level2','sf_d15_level2','sf_d16_level2','sf_d17_level2','sf_d18_level2','sf_d19_level2','sf_d20_level2',...
    'sf_d21_level2','sf_d22_level2','sf_d23_level2','sf_d24_level2','sf_d25_level2','sf_d26_level2','sf_d27_level2','sf_d28_level2','sf_d29_level2','sf_d30_level2',...
    'sf_d31_level2','sf_d32_level2','sf_d33_level2','sf_d34_level2','sf_d35_level2','sf_d36_level2','sf_d37_level2','sf_d38_level2','sf_d39_level2','sf_d40_level2',...
    'sf_d41_level2','sf_d42_level2','sf_d43_level2','sf_d44_level2','sf_d45_level2','sf_d46_level2','sf_d47_level2','sf_d48_level2','sf_d49_level2','sf_d50_level2',...
    'sf_d51_level2','sf_d52_level2','sf_d53_level2','sf_d54_level2','sf_d55_level2','sf_d56_level2','sf_d57_level2','sf_d58_level2','sf_d59_level2','sf_d60_level2',...
    'sf_d61_level2','sf_d62_level2','sf_d63_level2','sf_d64_level2'}, {'concat_out_level2'}) ;

%% Non-linear Mapping
net.addLayer('nonlinear2', Exponential_Params(), {'concat_out_level2'}, {'nonlinear_feature2'}, {'lambda2'});
f = net.getParamIndex('lambda2');
net.params(f).value = 0.5;
net.params(f).learningRate = 0;
net.params(f).weightDecay = 0.9;  

%% Max Pooling
net.addLayer('pooling_level2', ...
  dagnn.Pooling('poolSize', [2,2], 'pad', [0,1,0,1], 'stride', [1,1], 'method', 'max'), ...
  {'nonlinear_feature2'}, {'maxpool_feature2'});

%% Bilinear Upsampling
filters = single(bilinear_u(8, 1, level2_length));
net.addLayer('deconv2', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 4, ...
                      'crop', 2, ...
                      'hasBias', false), ...
             'maxpool_feature2', 'feature2', 'deconv2f');

f = net.getParamIndex('deconv2f');
net.params(f).value = filters;
net.params(f).learningRate = 0;
net.params(f).weightDecay = 1;

%% L2-normalization
net.addLayer('normalization_out2', l2norm(), {'feature2'}, {'FCSS_feature2'});

%% Level 3
%% L2-normalization
net.addLayer('normalization_level3', l2norm(), {'x17'}, {'feature_level3'});

%% Two-stream Shifting Transformer
for i = 1:level3_length
    sf = sprintf('sf_%.2d_level3', i);    
    sf_out = sprintf('sf_d%.2d_level3', i);
    
    sf_s_dx = sprintf('sf_s_dx%.2d_level3', i);
    sf_s_dy = sprintf('sf_s_dy%.2d_level3', i);
    sf_t_dx = sprintf('sf_t_dx%.2d_level3', i);
    sf_t_dy = sprintf('sf_t_dy%.2d_level3', i);
    
     net.addLayer(sf, CSSlayer(), {'feature_level3'}, {sf_out}, {sf_s_dx,sf_s_dy,sf_t_dx,sf_t_dy});
%     net.addLayer(sf, CSSlayer_shift(), {'feature_level3'}, {sf_out}, {sf_s_dx,sf_s_dy,sf_t_dx,sf_t_dy});
    
    f = net.getParamIndex(sf_s_dx);
    net.params(f).value = randn(1, 1, 'single')*5;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;
    f = net.getParamIndex(sf_s_dy);
    net.params(f).value = randn(1, 1, 'single')*5;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;
    f = net.getParamIndex(sf_t_dx);
    net.params(f).value = randn(1, 1, 'single')*5;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;
    f = net.getParamIndex(sf_t_dy);
    net.params(f).value = randn(1, 1, 'single')*5;
    net.params(f).learningRate = 0.1;
    net.params(f).weightDecay = 0.9;    
end

%% Concat. Network
net.addLayer('concat_level3', dagnn.Concat('dim',3),...
    {'sf_d01_level3','sf_d02_level3','sf_d03_level3','sf_d04_level3','sf_d05_level3','sf_d06_level3','sf_d07_level3','sf_d08_level3','sf_d09_level3','sf_d10_level3',...
    'sf_d11_level3','sf_d12_level3','sf_d13_level3','sf_d14_level3','sf_d15_level3','sf_d16_level3','sf_d17_level3','sf_d18_level3','sf_d19_level3','sf_d20_level3',...
    'sf_d21_level3','sf_d22_level3','sf_d23_level3','sf_d24_level3','sf_d25_level3','sf_d26_level3','sf_d27_level3','sf_d28_level3','sf_d29_level3','sf_d30_level3',...
    'sf_d31_level3','sf_d32_level3','sf_d33_level3','sf_d34_level3','sf_d35_level3','sf_d36_level3','sf_d37_level3','sf_d38_level3','sf_d39_level3','sf_d40_level3',...
    'sf_d41_level3','sf_d42_level3','sf_d43_level3','sf_d44_level3','sf_d45_level3','sf_d46_level3','sf_d47_level3','sf_d48_level3','sf_d49_level3','sf_d50_level3',...
    'sf_d51_level3','sf_d52_level3','sf_d53_level3','sf_d54_level3','sf_d55_level3','sf_d56_level3','sf_d57_level3','sf_d58_level3','sf_d59_level3','sf_d60_level3',...
    'sf_d61_level3','sf_d62_level3','sf_d63_level3','sf_d64_level3'}, {'concat_out_level3'});

%% Non-linear Mapping
net.addLayer('nonlinear3', Exponential_Params(), {'concat_out_level3'}, {'nonlinear_feature3'}, {'lambda3'});
f = net.getParamIndex('lambda3');
net.params(f).value = 0.5;
net.params(f).learningRate = 0;
net.params(f).weightDecay = 0.9;  

%% Max Pooling
net.addLayer('pooling_level3', ...
  dagnn.Pooling('poolSize', [2,2], 'pad', [0,1,0,1], 'stride', [1,1], 'method', 'max'), ...
  {'nonlinear_feature3'}, {'maxpool_feature3'});

%% Bilinear Upsampling
filters = single(bilinear_u(8, 1, level3_length));
net.addLayer('deconv3', ...
  dagnn.ConvTranspose('size', size(filters), ...
                      'upsample', 4, ...
                      'crop', 2, ...
                      'hasBias', false), ...
             'maxpool_feature3', 'feature3', 'deconv3f');

f = net.getParamIndex('deconv3f');
net.params(f).value = filters;
net.params(f).learningRate = 0;
net.params(f).weightDecay = 1;

%% L2-normalization
net.addLayer('normalization_out3', l2norm(), {'feature3'}, {'FCSS_feature3'});

%% Final Concat. Network
net.addLayer('concat_final', dagnn.Concat('dim',3), {'FCSS_feature1','FCSS_feature2','FCSS_feature3'}, {'FCSS_feature_concat'});

%% L2-normalization
net.addLayer('normalization_final', l2norm(), {'FCSS_feature_concat'},{'FCSS_feature'});

%% Dense Correspondene Loss
net.addLayer('objective', CorrespondenceLoss(), {'FCSS_feature','label'}, 'objective');

end
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.weights{2} = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
end