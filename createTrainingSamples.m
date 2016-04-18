function [trainingIndex, testingIndex, varargout] = createTrainingSamples(groundTruth, samplingRate, varargin)
% seperate the groundtruth into training and testing sampling based on the
% sampling rate
% count the regions in each class
% randomly create the seeds in each region, and then grow the seeds base on
% on the sampling rate in each region  
addpath('..\tools\RegionGrowing');
numofClass = max(groundTruth(:));
trainingIndex = cell(numofClass,1);
testingIndex = cell(numofClass,1);
seeds = cell(numofClass,1);
[m, n] = size(groundTruth);
for c = 1: numofClass
    classregion = (groundTruth == c); % get the region of a class
    if isempty(classregion)
        continue;
    end
    CC = bwconncomp(classregion, 8); % find the regions in a class
    numRegions = CC.NumObjects;
    indexSeeds = zeros(numRegions, 2);
    trainingRegion = cell(numRegions,1);
    for indexofRegion = 1 : numRegions
        regionIndex = CC.PixelIdxList{indexofRegion};
        numofTraingInRegion = round(numel(regionIndex)*samplingRate);
%         if numofTraingInRegion == 0 % if the regions multiply samplingRate is less than 1, then skip this region
%             trainingRegion{indexofRegion} = [];
%             continue;
%         end
        if nargin == 2
            randpoint = randi([1,numel(regionIndex)],1,1);
            [indexR, indexC] = ind2sub( [m,n], regionIndex(randpoint));
            indexSeeds(indexofRegion,:) = [indexR, indexC]; % save the seeds
        else 
            seeds = varargin{1};
            indexSeeds = seeds{c};
            indexR = indexSeeds(indexofRegion,1); 
            indexC = indexSeeds(indexofRegion,2);
        end
        regionClass = regiongrowingPixelNum(groundTruth, 1, [indexR,indexC], numofTraingInRegion);
        vregionClass = reshape(regionClass, [m*n,1]);
        trainingRegion{indexofRegion} = find(vregionClass == 1);
    end
    trainingIndex{c} = cell2mat(trainingRegion); 
    vtraingRegion = zeros(m*n,1); 
    vtraingRegion(trainingIndex{c}) = 1;
    vclassRegion = reshape(classregion, [m*n,1]);
    vtestregion = vclassRegion - vtraingRegion;
    testingIndex{c} = find(vtestregion == 1); 
    seeds{c} = indexSeeds;
end
if nargout == 3 
    varargout{1} = seeds;
end



