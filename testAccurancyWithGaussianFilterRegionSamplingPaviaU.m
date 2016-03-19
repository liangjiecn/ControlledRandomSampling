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
DataFile = 'PaviaU.mat';
addpath('..\data\remoteData');
addpath('..\tools\libsvm-3.20\matlab');
addpath('..\tools\RegionGrowing');
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
groundTruth = double(groundTruth);
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
stdlist = [-1:0.5:3];
dataCube = zeros(m,n,b);
for repeat = 1:10
    for i = 1 : length(sampleRateList)
        samplingRate = sampleRateList(i);
        if i == 1 % try to use the same seeds when using different sampling rate
            [trainingIndex, testingIndex, seeds] = createTrainingSamples(groundTruth, samplingRate);
        else
            [trainingIndex, testingIndex] = createTrainingSamples(groundTruth, samplingRate, seeds);
        end
        
         for indexofstd = 1:length(stdlist)
            stdgaussian = 2^stdlist(indexofstd);
            sizegaussian = floor((3*stdgaussian*2)/2)*2+1; % make sure the size is odd
            filter_mask = fspecial('gaussian',[sizegaussian, sizegaussian], stdgaussian);
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
resultsFile = ['Jresults\', mfilename, '.mat']; 
save(resultsFile, 'mu','sigma', 'accuracy' );
figure, plot(mu(1,:));
hold on
plot(mu(2,:), 'r');
plot(mu(3,:), 'g');
set(gca,'XTickLabel',{'0.50';'0.71'; '1.00'; '1.41'; '2.00';  '2.83'; '4.00'; '5.66'; '8.00'});
xlabel('Standard Deviation of the Gaussian Filter');
ylabel('Overall Classification Accuracy');
legend(' 5%', '10%', '25%');
figName = ['Jresults\', mfilename,'.fig']; 
hgsave(figName);


