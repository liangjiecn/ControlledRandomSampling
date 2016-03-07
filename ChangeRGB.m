% composite a false RGB image for remote sensing image

% DataFile = 'Indian_pines_corrected.mat';
% DataFile = 'Salinas_corrected.mat';
DataFile = 'Botswana.mat';
addpath('..\data\remote sensing data');
rawData = importdata(DataFile);% Load hyperspectral image and groud truth
datacube = normalise(rawData, 'percent',0.99);
% figure, imagesc(groundTruth);
[m, n, b] = size(datacube);
RGB = zeros(m, n, 3);
for i = 25:40 % 600:700
    for j = 15:25 % 500:600
        for k = 1: 15 % 400:500 
            sliceR = squeeze(datacube(:,:,i));
            sliceG = squeeze(datacube(:,:,j));
            sliceB = squeeze(datacube(:,:,k));
            RGB(:,:,1) = sliceR;
            RGB(:,:,2) = sliceG;
            RGB(:,:,3) = sliceB;
            imgname = sprintf('Botswana_R%dG%dB%d.png', i, j, k);
            imwrite(RGB, ['./temp/',imgname]);
        end
    end
end

