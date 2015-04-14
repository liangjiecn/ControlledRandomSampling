%run different methods on batch datasets with two sampling strategy
% to avoid out of memory problem, please incease the virtual memory
close all, 
clear, 
myCluster = parcluster('local');
myCluster.NumWorkers = 4;
saveProfile(myCluster);
numWorkers = matlabpool('size');
isPoolOpen = (numWorkers > 0);
if(~isPoolOpen)
    matlabpool;
end
% batch('..\data\remote sensing data', '*_gt.mat', @(x)NonlinearSVMSpatialFeatureRandomSampling(x,10));
% 
% batch('..\data\remote sensing data', '*.mat', @(x)NonlinearSVMSpectralFeatureRandomSampling(x, 10));
% batch('..\data\remote sensing data', '*.mat', @(x)LinearSVMSpectralFeatureRandomSampling(x, 10));
% batch('..\data\remote sensing data', '*.mat', @(x)NonlinearSVMSpectralFeatureContinuousSampling(x, 10));
% batch('..\data\remote sensing data', '*.mat', @(x)LinearSVMSpectralFeatureContinuousSampling(x, 10));

% batch('..\data\remote sensing data', '*.mat', @(x)LinearSVM3DDWTRandomSampling(x,10));
% batch('..\data\remote sensing data', '*.mat', @(x)LinearSVM3DDWTContinuousSampling(x,10));

% batch('..\data\remote sensing data', '*.mat', @(x)NonlinearSVMMorphologyRandomSampling(x,10));
 batch('..\data\remote sensing data', '*.mat', @(x)NonlinearSVM3DDWTRandomSampling(x,10));
 batch('..\data\remote sensing data', '*.mat', @(x)NonlinearSVM3DDWTContinuousSampling(x,10));
 

