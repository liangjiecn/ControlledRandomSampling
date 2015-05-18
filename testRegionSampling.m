function testRegionSampling(DataFile)
% test the region sampling with different sampling rate
% draw the original map, traing samples map and testing samples map
% DataFile: hyperspectral data file
addpath('..\data\remote sensing data');
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
dataCube = normalise(rawData, 'percent',1);
figure, imagesc(groundTruth);
[m, n, b] = size(rawData);
vdataCube =  reshape(dataCube, [m*n,b]);
numofClass = max(groundTruth(:));
trainingSamples = cell(numofClass,1);
testingSamples = cell(numofClass,1);
trainingLabels = cell(numofClass,1);
testingLabels = cell(numofClass,1);
numofTest = zeros(numofClass,1);
sampleRateList = [0.05, 0.1, 0.25];
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
    mtrainingLabels = cell2mat(trainingLabels);
    mtrainingIndex = cell2mat(trainingIndex);
    mtestingLabels = cell2mat(testingLabels);
    mtestingIndex = cell2mat(testingIndex); 
    trainingMap = zeros(m*n,1);
    trainingMap(mtrainingIndex) = mtrainingLabels;
    figure, imagesc(reshape(trainingMap,[m,n])); % check the training samples 
    testingMap = zeros(m*n,1);
    testingMap(mtestingIndex) = mtestingLabels;
    figure, imagesc(reshape(testingMap,[m,n]));
end
