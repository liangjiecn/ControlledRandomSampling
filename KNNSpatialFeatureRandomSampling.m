% hyperspectral classification with spatial feature(2d coordinates) using
% KNN
% Jie Liang
% 2015-05-20
function KNNSpatialFeatureRandomSampling(groundTruthFile, timeofRepeatition)
addpath('..\data\remote sensing data');
addpath('..\tools\libsvm-3.20\matlab');
groundTruth = importdata(groundTruthFile);
if ndims(groundTruth) > 2
    error('The dimension should be equal to 2');
end
resultsName = strrep(groundTruthFile, 'gt.mat', [mfilename, '.mat']);
resultsFile = ['Jresults\', resultsName]; 
% figure, imagesc(groundTruth);
[m, n] = size(groundTruth);
% produce the training samples by randomly selecting from the each class in ground truth 
vgroundTruth = reshape(groundTruth, [numel(groundTruth),1]);
numofClass = max(groundTruth(:));
trainingIndex = cell(numofClass,1);
testingIndex = cell(numofClass,1);
trainingSamples = cell(numofClass,1);
testingSamples = cell(numofClass,1);
trainingLabels = cell(numofClass,1);
testingLabels = cell(numofClass,1);
numofTest = zeros(numofClass,1);
% accuracyC = zeros(numofClass,3,10);
% accuracy = zeros(3,3);
sampleRateList = [0.05, 0.1, 0.25];
for repeat = 1:timeofRepeatition
for i = 1 : length(sampleRateList)
    sampleRate = sampleRateList(i);
    for c = 1: numofClass
        cc  = double(c);
        class = find(vgroundTruth == c);
        perm = randperm(numel(class)); 
        breakpoint = round(numel(class)*sampleRate);
        trainingIndex{c} = class(perm(1:breakpoint));
        testingIndex{c} = class(perm(breakpoint+1:end));
        [I, J] = ind2sub([m,n],trainingIndex{c});
        trainingSamples{c} = [I/m J/n];
%         trainingSamples{c} = [I, J];
        trainingLabels{c} = ones(length(trainingIndex{c}),1)*cc;
        [I, J] = ind2sub([m,n],testingIndex{c});
        testingSamples{c} = [I/m J/n];
%         testingSamples{c} =  [I, J];
        testingLabels{c} = ones(length(testingIndex{c}),1)*cc;
        numofTest(c) = numel(testingIndex{c});
    end
    mtrainingData = cell2mat(trainingSamples);
    mtrainingLabels = cell2mat(trainingLabels);
    mtrainingIndex = cell2mat(trainingIndex);
    mtestingData = cell2mat(testingSamples);
    mtestingLabels = cell2mat(testingLabels);
    mtestingIndex = cell2mat(testingIndex);  
    trainingMap = zeros(m*n,1);
    trainingMap(mtrainingIndex) = mtrainingLabels;
%     figure, imagesc(reshape(trainingMap,[m,n])); % check the training samples 
%    KNN
    mdl = ClassificationKNN.fit(mtrainingData,mtrainingLabels);
    predicted_labels = predict(mdl,mtestingData);
    
    resultMap = vgroundTruth;
    resultMap(mtestingIndex) = predicted_labels;
  %  figure, imagesc(reshape(resultMap,[m,n]));
    results(i, repeat) = assessment(mtestingLabels, predicted_labels, 'class' ); % calculate OA, kappa, AA
end
end
save(resultsFile, 'results');

