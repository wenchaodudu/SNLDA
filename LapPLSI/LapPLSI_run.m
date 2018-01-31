%% load data
import = load('../20-news-data/ng2011293x8165itrn.mat');
% fieldnames(import)
Dtrn = import.Dtrn;
% fieldnames(Dtrn)
data = Dtrn.data; % training data matrix
labels = Dtrn.labels; % label matrix
%% run only once
mex mex_Pw_d.c
mex mex_EMstep.c
mex mex_logL.c
%% constructing neighborhood graph
% inputs of main function
% [Pz_d_final, Pw_z_final, Obj_final, nIter_final] = LapPLSI(X, K, W, options, Pz_d, Pw_z)
data_subset = data(1:100,1:200);
K = 10; % number of neighbors
W = weightMat(data_subset, K); % weight matrix of the affinity graph
%% run model
LapPLSIoptions = [];
LapPLSIoptions.WeightMode = 'Cosine';
LapPLSIoptions.bNormalized = 1;
LapPLSIoptions.k = 7;
LapPLSIoptions.maxIter = 100;
LapPLSIoptions.lambda = 100;
LapPLSIoptions.Verbosity = 1;
nTopics = 10;
[Pz_d_final, Pw_z_final, Obj_final, nIter_final, Pz_d_init,Pw_z_init] = LapPLSI(data_subset', nTopics, W, LapPLSIoptions);
