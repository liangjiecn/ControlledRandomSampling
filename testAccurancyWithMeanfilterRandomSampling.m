% test the classification rate with the size of filter
% use mean filter to cover the spatial information
close all 
clear,
myCluster = parcluster('local');
myCluster.NumWorkers = 6;
saveProfile(myCluster);
numWorkers = matlabpool('size');
isPoolOpen = (numWorkers > 0);
if(~isPoolOpen)
    matlabpool;
end
DataFile = 'Indian_pines_corrected.mat';
addpath('..\data\remote sensing data');
addpath('..\tools\libsvm-3.20\matlab');
resultsFile = ['Jresults\', mfilename]; 
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
groundTruth = importdata([subfix, '_gt.mat']);
[m, n, b] = size(rawData); 
vgroundTruth = reshape(groundTruth, [numel(groundTruth),1]);
numofClass = max(groundTruth(:));
trainingIndex = cell(numofClass,1);
testingIndex = cell(numofClass,1);
trainingSamples = cell(numofClass,1);
testingSamples = cell(numofClass,1);
trainingLabels = cell(numofClass,1);
testingLabels = cell(numofClass,1);
numofTest = zeros(numofClass,1);
% accuracyC = zeros(numofClass,3);
sampleRateList = [0.05, 0.1, 0.25];
filterSizeList = [1 3, 5, 7, 9, 11];
dataCube = zeros(m,n,b);
for repeat = 1:10 % repeat 10 times
    for i = 1 : length(sampleRateList)
        sampleRate = sampleRateList(i);
        for c = 1: numofClass
            cc  = double(c);
            class = find(vgroundTruth == c);
            if isempty(class)
                continue;
            end
            perm = randperm(numel(class));  %  random sampling
            breakpoint = round(numel(class)*sampleRate);
            trainingIndex{c} = class(perm(1:breakpoint));
            testingIndex{c} = class(perm(breakpoint+1:end));
            numofTest(c) = numel(testingIndex{c});
        end
        for indexofSize = 1:length(filterSizeList)
            filterSize = filterSizeList(indexofSize);
            filter_mask=1/(filterSize*filterSize)*ones(filterSize,filterSize);
            for j = 1:size(rawData,3)
                dataCube(:,:,j) = conv2( rawData(:,:,j),filter_mask,'same');
            end 
            dataCube = normalise(dataCube,'percent', 1);
            vdataCube = reshape(dataCube,[m*n,b]);
            for c = 1: numofClass
                cc  = double(c);
                trainingSamples{c} = vdataCube(trainingIndex{c},:);
                trainingLabels{c} = ones(length(trainingIndex{c}),1)*cc;
                testingSamples{c} = vdataCube(testingIndex{c},:);
                testingLabels{c} = ones(length(testingIndex{c}),1)*cc;
            end

            mtrainingData = cell2mat(trainingSamples);
            mtrainingLabels = cell2mat(trainingLabels);
            mtrainingIndex = cell2mat(trainingIndex);
            mtestingData = cell2mat(testingSamples);
            mtestingLabels = cell2mat(testingLabels);
            mtestingIndex = cell2mat(testingIndex); 
            trainingMap = zeros(m*n,1);
            trainingMap(mtrainingIndex) = mtrainingLabels;
    %       figure; imagesc(reshape(trainingMap,[m,n])); % check the training samples 
            mtrainingData = double(mtrainingData);
            %select parameters c and g
            log2cList = -1:1:16;
            cv = zeros(length(log2cList), 1);
            parfor indexC = 1:length(log2cList)
                log2c = log2cList(indexC);
                cmd = ['-q -t 0 -v 5 -c ', num2str(2^log2c)];
                cv(indexC) = svmtrain(mtrainingLabels, mtrainingData, cmd);
            end
            [~, indexcv]= max(cv);
            bestc = 2^log2cList(indexcv); 
            optPara = [ '-q -t 0 -c ', num2str(bestc)];
            svm = svmtrain(mtrainingLabels, mtrainingData,  optPara);   
            mtestingData = double(mtestingData);
            [predicted_label, rr, prob_estimates] = svmpredict(mtestingLabels, mtestingData, svm);  
            accuracy(i,indexofSize,repeat) = rr(1);
            resultMap = vgroundTruth;
            resultMap(mtestingIndex) = predicted_label;
%             figure; imagesc(reshape(resultMap,[m,n]));
            % accurancy in each class
%             resultC = predicted_label == mtestingLabels;
%             for c = 1:numofClass
%                 accuracyC(c,i) = sum(resultC(find(mtestingLabels == c)))/numofTest(c);
%             end
    %         save(resultsFile, 'accuracy', 'accuracyC' );
        end
    end
end

mu = mean(accuracy,3); sigma = std(accuracy,0, 3);
save(resultsFile, 'mu','sigma','accuracy' );
figure, plot(mu(1,:));
hold on
plot(mu(2,:), 'r');
plot(mu(3,:), 'g');
set(gca,'XTickLabel',{'';'1*1'; '3*3'; '5*5'; '7*7';  '9*9'; '11*11'});




