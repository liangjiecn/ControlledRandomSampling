% hyperspectral classification with spatial feature(2d coordinates) using random sampling
% and nonlinear SVM
% Jie Liang
% 2014-09-12
function NonlinearSVMSpatialFeatureRandomSampling(groundTruthFile, timeofRepeatition)
addpath('..\data\remote sensing data');
addpath('..\tools\libsvm-3.20\matlab');
groundTruth = importdata(groundTruthFile);
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
%     select parameters c and g
    log2cList = -5:1:6;
    log2gList = -5:1:6;
    cv = zeros(length(log2cList), length(log2gList) );
    parfor indexC = 1:length(log2cList)
        log2c = log2cList(indexC);
        tempcv = zeros(1,length(log2gList));
        for indexG = 1:length(log2gList)
           log2g =  log2gList(indexG);
           cmd = ['-q -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
           tempcv(indexG) = svmtrain(mtrainingLabels, mtrainingData, cmd);
        end
        cv(indexC,:) = tempcv;
    end
    [~, indexcv]= max(cv(:));
    [bestindexC, bestindexG] = ind2sub(size(cv), indexcv);
    bestc = 2^log2cList(bestindexC);
    bestg = 2^log2gList(bestindexG);
    optPara = [ '-q -c ', num2str(bestc), ' -g ', num2str(bestg)];
    svm = svmtrain(mtrainingLabels, mtrainingData, optPara);    % [, 'libsvm_options']);
    [predicted_labels, ~, ~] = svmpredict(mtestingLabels, mtestingData, svm);  
    resultMap = vgroundTruth;
    resultMap(mtestingIndex) = predicted_labels;
%   figure, imagesc(reshape(resultMap,[m,n]));
    results(i, repeat) = assessment(mtestingLabels, predicted_labels, 'class' ); % calculate OA, kappa, AA
end
end
save(resultsFile, 'results');

