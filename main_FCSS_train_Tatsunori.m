%% Script for learning FCSS descriptor network
%% Written by Seungryong Kim, Yonsei University, Seoul, Korea
function main_FCSS_train_Tatsunori(varargin)
    run('vlfeat-0.9.20\toolbox\vl_setup');
    run('matconvnet-1.0-beta23\matlab\vl_setupnn.m');
    addpath('SIFTflow');
    addpath('flow-code-matlab');
    addpath('init_model');
    addpath('model');
    addpath('function');
    
    load('data/imdb_correspondence_Tatsunori.mat');

    init_model = true; % Using pretrained model as an initial parameter (or not)
    net = init_FCSS(init_model);

    trainOpts.batchSize = 2;
    trainOpts.numEpochs = 600;
    trainOpts.continue = true;
    trainOpts.gpus = 1;
    trainOpts.learningRate = 1e-3;
    trainOpts.derOutputs = {'objective', 1};
    trainOpts.expDir = 'data/fcss_Tatsunori';
    trainOpts = vl_argparse(trainOpts, varargin);

    cnn_train_dag_pairwise_learning(net, imdb, getBatch, trainOpts);    
end

function inputs = getBatch()
    inputs = @(imdb,batch) getBatch_Tatsunori(imdb,batch);
end
