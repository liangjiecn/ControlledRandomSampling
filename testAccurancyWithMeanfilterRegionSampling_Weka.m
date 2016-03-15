% test the classification rate with the size of filter
% use mean filter to cover the spatial information
close all 
clear,
DataFile = 'Indian_pines_corrected.mat';
addpath('..\data\remoteData');
addpath('..\tools\export_fig');
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
sampleRateList = [0.05, 0.1, 0.25];
filterSizeList = 1:2:27;
dataCube = zeros(m,n,b);
for repeat = 1:10 % repeat 10 times
    for i = 1 : length(sampleRateList)
        samplingRate = sampleRateList(i);
        if i == 1 % try to use the same seeds when using different sampling rate
            [trainingIndex, testingIndex, seeds] = createTrainingSamples(groundTruth, samplingRate);
        else
            [trainingIndex, testingIndex] = createTrainingSamples(groundTruth, samplingRate, seeds);
        end
        
        for indexofSize = 1:length(filterSizeList)
            filterSize = filterSizeList(indexofSize);
            filter_mask=1/(filterSize*filterSize)*ones(filterSize,filterSize);
            for j = 1:size(rawData,3)
                dataCube(:,:,j)=conv2( rawData(:,:,j),filter_mask,'same');
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
%           classification
            predicted_labels = wekaClassificationWarp(mtrainingData, mtrainingLabels, mtestingData);  
            results = assessment(mtestingLabels, predicted_labels, 'class' ); % calculate OA, kappa, AA    
            accuracy(i,indexofSize,repeat) = results.OA;
            resultMap = vgroundTruth;
            resultMap(mtestingIndex) = predicted_labels;
%           figure; imagesc(reshape(resultMap,[m,n]));   
%             axis image,
%             axis off,
%             figName = ['Jresults\maps\', mfilename,'_sampling' num2str(i), '_std', num2str(indexofSize), '.fig']; 
%             hgsave(figName);
%             hgload(figName);
%             setImage('Jresults\maps');

        end
    end
end
mu = mean(accuracy,3); sigma = std(accuracy,0, 3);
resultsFile = ['Jresults\', mfilename, '.mat']; 
save(resultsFile, 'mu','sigma', 'accuracy' );
figure, plot(1:14, mu(1,:));
hold on
plot(1:length(filterSizeList),lengthmu(2,:), 'r');
plot(mu(3,:), 'g');
set(gca,'XLim',[1 14]);
set(gca,'XTick',1:27);
set(gca,'XTickLabel',{'1'; '3'; '5'; '7';  '9'; '11'; '13'; ...
                      '15'; '17';'19'; '21'; '23';'25';'27'});
xlabel('Size of the Mean Filter');
ylabel('Overall Classification Accuracy');
legend(' 5%', '10%', '25%');
figName = ['Jresults\', mfilename,'.fig']; 
hgsave(figName);





