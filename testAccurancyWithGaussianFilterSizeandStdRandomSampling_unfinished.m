% test the classification rate with the size of filter
% use mean filter to cover the spatial information
close all 
clear,
myCluster = parcluster('local');
myCluster.NumWorkers = 8;
saveProfile(myCluster);
numWorkers = matlabpool('size');
isPoolOpen = (numWorkers > 0);
if(~isPoolOpen)
    matlabpool;
end
DataFile = 'Indian_pines_corrected.mat';
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
stdlist = [-1:0.5:2];
sizelist = [1, 3, 5, 7, 9, 11];
dataCube = zeros(m,n,b);
for repeat = 1:10
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
          %  numofTest(c) = numel(testingIndex{c});
        end
        
         for indexofstd = 1:length(stdlist)
            stdgaussian = 2^stdlist(indexofstd);
            filter_mask = fspecial('gaussian',[9, 9], stdgaussian);
            for j = 1:size(rawData,3)
                dataCube(:,:,j) = imfilter(rawData(:,:,j), filter_mask);
            end 
            dataCube = normalise(dataCube,'percent', 1);
            vdataCube = reshape(dataCube,[m*n,b]);                    
            mtrainingIndex = cell2mat(trainingIndex);
            mtrainingData = vdataCube(mtrainingIndex,:);
            mtrainingLabels = vgroundTruth(mtrainingIndex);
            trainingMap = zeros(m*n,1);
            trainingMap(mtrainingIndex) = mtrainingLabels;
%           figure; imagesc(reshape(trainingMap,[m,n])); % check the training samples 
            mtestingIndex = cell2mat(testingIndex);
            mtestingData =  vdataCube(mtestingIndex,:);
            mtestingLabels = vgroundTruth(mtestingIndex);    
            testingMap = zeros(m*n,1);
            testingMap(mtestingIndex) = mtestingLabels;
%           figure; imagesc(reshape(testingMap,[m,n])); % check the training samples        
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
            [predicted_label, ac, ~] = svmpredict(mtestingLabels, mtestingData, svm);  
            accuracy(i,indexofstd,repeat) = ac(1);
            resultMap = vgroundTruth;
            resultMap(mtestingIndex) = predicted_label;
%           figure; imagesc(reshape(resultMap,[m,n]));
        end
    end
end

mu = mean(accuracy,3); sigma = std(accuracy,0, 3);
save('Jresults\testAccurancyWithGaussianFilterRandomSampling.mat', 'mu','sigma',...
    'accuracy' );
figure, plot(mu(1,:));
hold on
plot(mu(2,:), 'r');
plot(mu(3,:), 'g');
title('Classification accuracy using Gassian filter with different standard deviation.')


