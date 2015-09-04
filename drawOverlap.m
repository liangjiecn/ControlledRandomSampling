% calculate and draw the overlap of training samples and testing samples
close all,
addpath('..\data\remote sensing data');
groundTruth = importdata('Indian_gt.mat');
[m, n] = size(groundTruth);
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
        trainingLabels{c} = ones(length(trainingIndex{c}),1)*cc;
        testingLabels{c} = ones(length(testingIndex{c}),1)*cc;
        numofTest(c) = numel(testingIndex{c});
    end
    mtrainingLabels = cell2mat(trainingLabels);
    mtrainingIndex = cell2mat(trainingIndex);
    mtestingLabels = cell2mat(testingLabels);
    mtestingIndex = cell2mat(testingIndex); 
    trainingMap = zeros(m*n,1);
    trainingMap(mtrainingIndex) = mtrainingLabels;
    tempGroundTruth = padarray(groundTruth,[1,1]); % incase for the Subscript out of border
    tempGroundTruth(tempGroundTruth>0) = 1;
    window = ones(3,3);
    window(2,2) = 0;
    overlap = 0;
    figure, h = imagesc(groundTruth);
    axis image;
    hold on,
    for j = 1: size(mtrainingIndex)   
        [x,y] = ind2sub([m,n],mtrainingIndex(j));
        plot(y,x,'w.' )
        tempGroundTruth(x:x+2, y:y+2) = 0;
    end  
    leftTest = sum(tempGroundTruth(:));
    percentage = 1 - leftTest/numel(mtestingIndex);
    title(['Overlap = ', num2str(percentage*100), '% when sampling rate = ', num2str(sampleRate*100), '%'], 'FontSize', 18);
    hold off;
end


