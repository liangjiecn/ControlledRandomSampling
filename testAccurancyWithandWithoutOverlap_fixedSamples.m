% sampling fixed number of samples as training data
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
addpath('..\data\remoteData');
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
sampleNumList = [3, 5, 10, 20];
filterSizeList = 1:2:27;
dataCube = zeros(m,n,b);
for repeat = 1:10
    for i = 1 : length(sampleNumList)
        sampleNum = sampleNumList(i);
        for c = 1: numofClass
            cc  = double(c);
            class = find(vgroundTruth == c);
            if isempty(class)
                continue;
            end
            perm = randperm(numel(class));  %  random sampling
            breakpoint = sampleNum;
            trainingIndex{c} = class(perm(1:breakpoint));
            testingIndex{c} = class(perm(breakpoint+1:end)); 
        end
        for indexofSize = 1:length(filterSizeList)
            filterSize = filterSizeList(indexofSize);
            filter_mask=1/(filterSize*filterSize)*ones(filterSize,filterSize);
            for j = 1:size(rawData,3)
                dataCube(:,:,j)=conv2( rawData(:,:,j),filter_mask,'same');
            end 
            dataCube = normalise(dataCube,'percent', 1);
            vdataCube = reshape(dataCube,[m*n,b]);                    
            mtrainingIndex = cell2mat(trainingIndex);
            mtrainingData = vdataCube(mtrainingIndex,:);
            mtrainingLabels = vgroundTruth(mtrainingIndex);
            trainingMap = zeros(m*n,1);
            trainingMap(mtrainingIndex) = mtrainingLabels;
%             figure; imagesc(reshape(trainingMap,[m,n])); % check the training samples 
            mtestingIndex = cell2mat(testingIndex);
            mtestingData =  vdataCube(mtestingIndex,:);
            mtestingLabels = vgroundTruth(mtestingIndex);    
            % find non-overlapping testing data
            halfheightfilter = floor(filterSize/2);
            halfwidthfilter = floor(filterSize/2);
            tempGroundTruth = padarray(groundTruth,[halfheightfilter,halfwidthfilter]); % incase for the Subscript out of border
            tempGroundTruth(tempGroundTruth>0) = 1; % assign all samples to non zero
            for indexpoint = 1: size(mtrainingIndex)   
                [x,y] = ind2sub([m,n],mtrainingIndex(indexpoint));
                tempGroundTruth(x:x+2*halfheightfilter, y:y+2*halfwidthfilter) = 0; % assign the training area to zero
            end  
            leftTest = sum(tempGroundTruth(:));
            percentage(i,indexofSize,repeat) = 1 - leftTest/numel(mtestingIndex);
%             disp(percentage);
            % remove the padded edges
            tempGroundTruth = ...
                tempGroundTruth(halfheightfilter+1:end-halfheightfilter, halfwidthfilter+1: end-halfwidthfilter);% recover the real size
            nonmtestingIndex = find(tempGroundTruth == 1); %find the left testing samples (non-overlap)
            nonmtestingData =  vdataCube(nonmtestingIndex,:);
            nonmtestingLabels = vgroundTruth(nonmtestingIndex); 
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
            [predicted_label, ac, ~] = svmpredict(mtestingLabels, mtestingData, svm);% classify all testing samples
            accuracy(i,indexofSize,repeat) = ac(1);
            resultMap = vgroundTruth;
            resultMap(mtestingIndex) = predicted_label;
            nonmtestingData = double(nonmtestingData);
            [predicted_label, ac, ~] = svmpredict(nonmtestingLabels, nonmtestingData, svm); % classify non-overlap testing samples
            nonaccuracy(i,indexofSize,repeat) = ac(1);
%             figure; imagesc(reshape(resultMap,[m,n]));
        end
    end
end
mu = mean(accuracy,3); sigma = std(accuracy,0, 3);
nonmu = mean(nonaccuracy,3); nonsigma = std(nonaccuracy,0, 3);
mp = mean(percentage, 3); sigmap = std(percentage,0, 3);
resultsFile = ['Jresults\', mfilename, '.mat']; 
save(resultsFile, 'mu','sigma',...
    'accuracy','nonmu','nonsigma','nonaccuracy','percentage' );
figure, plot(mu(1,:));
hold on
plot(mu(2,:), 'r');
plot(mu(3,:), 'g');
plot(mu(4,:), 'c');
set(gca,'XLim',[1 14]);
set(gca,'XTick',1:27);
set(gca,'XTickLabel',{'1'; '3'; '5'; '7';  '9'; '11'; '13'; ...
                      '15'; '17';'19'; '21'; '23';'25';'27'});
xlabel('Size of the Mean Filter');
ylabel('Overall Classification Accuracy');
legend('3', '5', '10', '20');
figName = ['Jresults\', mfilename,'_Accuracy.fig']; 
hgsave(figName);

figure, plot(mp(1,:));
hold on
plot(mp(2,:), 'r');
plot(mp(3,:), 'g');
plot(mp(4,:), 'c');
set(gca,'XLim',[1 14]);
set(gca,'XTick',1:27);
set(gca,'XTickLabel',{'1'; '3'; '5'; '7';  '9'; '11'; '13'; ...
                      '15'; '17';'19'; '21'; '23';'25';'27'});
xlabel('Size of the Mean Filter');
ylabel('Percentage of Overlap');
legend('3', '5', '10', '20');
figName = ['Jresults\', mfilename,'_overlap.fig']; 
hgsave(figName);






