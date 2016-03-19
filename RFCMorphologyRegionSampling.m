function RFCMorphologyRegionSampling(DataFile, timeofRepeatition)
% hyperspectral classification with spectral feature using random sampling
% and nonlinear SVM
addpath('..\data\remoteData');
addpath('..\tools\libsvm-3.20\matlab');
addpath('..\tools\RegionGrowing');
addpath('..\tools\matlab2weka');
rawData = importdata(DataFile);% Load hyperspectral image and groud truth
if ndims(rawData) ~= 3
    return;
end
indexof_= find(DataFile == '_',1);
if isempty(indexof_)
    subfix = DataFile(1:end-4);
else
    subfix = DataFile(1:indexof_-1);
end
resultsFile = ['Jresults\', subfix, '_', mfilename, '.mat']; 
groundTruth = importdata([subfix, '_gt.mat']);
dataCube = mm(rawData);
% figure, imagesc(groundTruth);
[m, n, b] = size(dataCube);
vdataCube =  reshape(dataCube, [m*n,b]);
vgroundTruth = reshape(groundTruth, [numel(groundTruth),1]);
numofClass = max(groundTruth(:));
trainingSamples = cell(numofClass,1);
testingSamples = cell(numofClass,1);
trainingLabels = cell(numofClass,1);
testingLabels = cell(numofClass,1);
numofTest = zeros(numofClass,1);
sampleRateList = [0.05, 0.1, 0.25];
for repeat = 1:timeofRepeatition
for i = 1 : length(sampleRateList)
    samplingRate = sampleRateList(i);
    if i == 1 % try to use the same seeds when using different sampling rate
        [trainingIndex, testingIndex, seeds] = createTrainingSamples(groundTruth, samplingRate);
    else
        [trainingIndex, testingIndex] = createTrainingSamples(groundTruth, samplingRate, seeds);
    end
    for c = 1: numofClass
        cc  = double(c);
        trainingSamples{c} = vdataCube(trainingIndex{c},:);
        trainingLabels{c} = ones(length(trainingIndex{c}),1)*cc;
        testingSamples{c} = vdataCube(testingIndex{c},:);
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
%   figure, imagesc(reshape(trainingMap,[m,n])); % check the training samples 
    mtrainingData = double(mtrainingData);
%   classification
	predicted_labels = wekaClassificationWarp(mtrainingData, mtrainingLabels, mtestingData);  
	results = assessment(mtestingLabels, predicted_labels, 'class' ); % calculate OA, kappa, AA    
	accuracy(i, repeat) = results.OA;
	resultMap = vgroundTruth;
	resultMap(mtestingIndex) = predicted_labels;
%figure; imagesc(reshape(resultMap,[m,n]));
end
end
save(resultsFile, 'results');