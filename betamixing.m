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
lx = 9; % window length
ly = 9; % window width
hlx = floor(lx/2); % half
hly = floor(ly/2);
count = 1;
scoTemp = zeros(m*n, lx*ly); % statistical correlation
for i = 1:m
    if i-hlx <=0 || i+hlx >m
        continue;
    end
    for j = 1:n
        if j-hly <= 0 || j+hly > n
            continue;
        end
        vc = squeeze(rawData(i, j, :));
        neighbourTemp = rawData(i-hlx:i+hlx, j-hly:j+hly, :);
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

figure,
imshow(sco, []);
colormap(hsv);
figure,
mesh(sco);