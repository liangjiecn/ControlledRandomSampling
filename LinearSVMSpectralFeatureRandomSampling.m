function LinearSVMSpectralFeatureRandomSampling(DataFile, timeofRepeatition)
% hyperspectral classification with spectral feature using random sampling
% and linear SVM
addpath('..\data\remote sensing data');
addpath('..\tools\libsvm-3.20\matlab');
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
resultsFile = ['results\', subfix, '_LinearSVMSpectralFeatureRandomSampling.mat']; 
groundTruth = importdata([subfix, '_gt.mat']);
figure, imagesc(groundTruth);
[m, n, b] = size(rawData);
dataCube = mapstd(rawData);
vdataCube =  reshape(dataCube, [m*n,b]);
vgroundTruth = reshape(groundTruth, [numel(groundTruth),1]);
numofClass = max(groundTruth(:));
trainingIndex = cell(numofClass,1);
testingIndex = cell(numofClass,1);
trainingSamples = cell(numofClass,1);
testingSamples = cell(numofClass,1);
trainingLabels = cell(numofClass,1);
testingLabels = cell(numofClass,1);
numofTest = zeros(numofClass,1);
sampleRateList = [0.05, 0.1, 0.25];

for repeat = 1:timeofRepeatition
for i = 1 : length(sampleRateList)
    sampleRate = sampleRateList(i);
    for c = 1: numofClass
        cc  = double(c);
        class = find(vgroundTruth == c);
        if isempty(class)
            continue;
        end
        perm = randperm(numel(class)); 
        breakpoint = round(numel(class)*sampleRate);
        trainingIndex{c} = class(perm(1:breakpoint));
        testingIndex{c} = class(perm(breakpoint+1:end));
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
%     figure, imagesc(reshape(trainingMap,[m,n])); % check the training samples 
    mtrainingData = double(mtrainingData);
    %select parameters c and g
    log2cList = -1:0.5:5;
    cv = zeros(length(log2cList), 1);
    parfor indexC = 1:length(log2cList)
        log2c = log2cList(indexC);
        cmd = ['-q -t 0 -v 5 -c ', num2str(2^log2c)];
        cv(indexC) = svmtrain(mtrainingLabels, mtrainingData, cmd);
    end
    [~, indexcv]= max(cv);
    bestc = 2^log2cList(indexcv); 
    optPara = [ '-q -t 0 -c ', num2str(bestc)];
    svm = svmtrain(mtrainingLabels, mtrainingData, optPara);   
    mtestingData = double(mtestingData);
    [predicted_label, rr, prob_estimates] = svmpredict(mtestingLabels, mtestingData, svm);  
    accuracy(i, repeat) = rr(1);
    resultMap = vgroundTruth;
    resultMap(mtestingIndex) = predicted_label;
%     figure, imagesc(reshape(resultMap,[m,n]));
    % accurancy in each class
    resultC = predicted_label == mtestingLabels;
    for c = 1:numofClass
        accuracyC(c,i,repeat) = sum(resultC(find(mtestingLabels == c)))/numofTest(c);
    end  
end
end
mu = mean(accuracy,2);
sigma = std(accuracy, 0, 2);
save(resultsFile, 'mu','sigma','accuracy', 'accuracyC' );