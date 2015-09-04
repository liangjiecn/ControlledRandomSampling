% measure the correlation of spectral response between a single pixel and
% its neighbouring location
% check if the dependence decreases with the distance so that it meats the
% requirement of Beta mixing

% statistical correlation across the full map 
close all 
clear,
DataFile = 'Indian_pines_corrected.mat';
addpath('..\data\remote sensing data');
addpath('..\tools\libsvm-3.20\matlab');
rawData = importdata(DataFile);% Load hyperspectral image and groud truth
[m, n, b] = size(rawData);
groundTruth = importdata('Indian_gt.mat');

% filter parameters
filterSizeList = [1 3, 5, 7, 9, 11];

lx = 15; % window length
ly = 15; % window width
hlx = floor(lx/2); % half
hly = floor(ly/2);
count = 1;
scoTemp = zeros(m*n, lx*ly); % statistical correlation
scofilter = zeros(length(filterSizeList), lx,ly);
figure,
hold all,
for indexofSize = 1:length(filterSizeList)
    filterSize = filterSizeList(indexofSize);
    filter_mask=1/(filterSize*filterSize)*ones(filterSize,filterSize);
    for j = 1:size(rawData,3)
        dataCube(:,:,j)=conv2( rawData(:,:,j),filter_mask,'same');
    end

    for i = 1:m
        if i-hlx <=0 || i+hlx >m
            continue;
        end
        for j = 1:n
            if j-hly <= 0 || j+hly > n
                continue;
            end
            vc = squeeze(dataCube(i, j, :));
            neighbourTemp = dataCube(i-hlx:i+hlx, j-hly:j+hly, :);
            neighbourTemp = reshape(neighbourTemp,[lx*ly,b]);
            neighbour = neighbourTemp';
            co = corr(vc, neighbour);
            scoTemp(count,:) = co;
            count = count + 1;
        end 
    end
    sco = scoTemp(1:count-1,:);
    sco = mean(sco);
    sco = reshape(sco,[lx,ly]);
    scofilter(indexofSize,:,:) = sco;
    plot(0:7, sco(5,8:end));
end
