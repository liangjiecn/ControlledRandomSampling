close all, 
clear, 
% batch('..\data\remote sensing data', '*_gt.mat', @(x)NonlinearSVMSpatialFeatureRandomSampling(x,1));

% batch('..\data\remote sensing data', '*.mat', @(x)NonlinearSVMSpectralFeatureRandomSampling(x, 1));
% batch('..\data\remote sensing data', '*.mat', @(x)LinearSVMSpectralFeatureRandomSampling(x, 1));
% batch('..\data\remote sensing data', '*.mat', @(x)NonlinearSVMSpectralFeatureContinuousSampling(x, 1));
% batch('..\data\remote sensing data', '*.mat', @(x)LinearSVMSpectralFeatureContinuousSampling(x, 1));

% batch('..\data\remote sensing data', '*.mat', @(x)LinearSVM3DDWTRandomSampling(x,1));
% batch('..\data\remote sensing data', '*.mat', @(x)LinearSVM3DDWTContinuousSampling(x,1));


 batch('..\data\remote sensing data', '*.mat', @(x)NonlinearSVMMorphologyRandomSampling(x,1));