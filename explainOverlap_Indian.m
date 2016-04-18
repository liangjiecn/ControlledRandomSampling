% draw the overlap under two sampling strategies
clear, close all,
addpath('..\data\remoteData');
addpath('..\tools\export_fig');
subfix = 'Indian';
groundTruth = importdata([subfix, '_gt.mat']);
[m, n, b] = size(groundTruth);
figure, imagesc(groundTruth), axis image, axis off;
cmap = importdata([subfix, '_colorMap.mat']);

% draw sampling regions
figure, h = hgload('Jresults/Indian_regionsampling5.fig');
I_region = getimage(h);
h = figure, imshow(uint8(I_region), cmap);
hold on;
% draw boundries
gmap = groundTruth;
gmap(gmap>0) = 1;
gmap = logical(gmap);
[B] = bwboundaries(gmap);
for k=1:length(B),
    boundary = B{k};
    plot(boundary(:,2),...
            boundary(:,1),'k','LineStyle', '--','LineWidth',2);
end
Position = [200 0 1200 900];
set(h,'Position', Position);
savefig('Jresults/Indian_region.fig');
hgload('Jresults/Indian_region.fig');
setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')
% draw sampling points
figure, h = hgload('Jresults/Indian_randomsampling5.fig');
I_random = getimage(h);
h = figure, imshow(uint8(I_random), cmap);
hold on;
% draw boundries
for k=1:length(B),
    boundary = B{k};
    plot(boundary(:,2),...
            boundary(:,1),'k','LineStyle', '--','LineWidth',2);
end
Position = [200 0 1200 900];
set(h,'Position', Position);
savefig('Jresults/Indian_random.fig');
hgload('Jresults/Indian_random.fig');
setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')
% filter 
background  = 1;
foreground = 0;
bIdx = I_region == 0;
fIdx = I_region >0;
I_region(bIdx) = background;
I_region(fIdx) = foreground;
filter = fspecial('gaussian', [5, 5], 2);
SI_region = imfilter(I_region, filter, 'replicate');
h = figure; imshow(SI_region);
hold on;
% draw boundries
for k=1:length(B),
    boundary = B{k};
    plot(boundary(:,2),...
            boundary(:,1),'k','LineStyle', '--','LineWidth',2);
end
Position = [200 0 1200 900];
set(h,'Position', Position);
savefig('Jresults/Indian_region_filter.fig');
hgload('Jresults/Indian_region_filter.fig');
setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')

bIdx = I_random == 0;
fIdx = I_random >0;
I_random(bIdx) = background;
I_random(fIdx) = foreground;
ST_random = imfilter(I_random, filter, 'replicate');
h = figure; imshow(ST_random,[]);
hold on, 
% draw boundries
for k=1:length(B),
    boundary = B{k};
    plot(boundary(:,2),...
            boundary(:,1),'k','LineStyle', '--','LineWidth',2);
end
Position = [200 0 1200 900];
set(h,'Position', Position);
savefig('Jresults/Indian_random_filter.fig');
hgload('Jresults/Indian_random_filter.fig');
setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')