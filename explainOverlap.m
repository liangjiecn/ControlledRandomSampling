% draw the overlap under two sampling strategies
clear, close all,
background  = 1;
foreground = 0;
% nrow = 128; ncol = 152;
nrow = 65; ncol = 77;
halfrow = floor(nrow/2); halfcol = floor(ncol/2);
midpoint = [halfrow+1, halfcol+1];
J = ones(nrow,ncol)*background;
samplingRate = 0.05;
numofTraining = floor(nrow * ncol * samplingRate);
temp = round(sqrt(numofTraining));
width = round(temp/2)*2+1;
height = width;
indexWidth = [midpoint(1)-floor(width/2) : midpoint(1) + floor(width/2)];
indexHight = [midpoint(2)-floor(height/2) : midpoint(2)+floor(height/2)];
T_region = J;
T_region(round(indexHight),round(indexWidth)) = foreground;

h = figure; 
imshow(T_region); % check the training samples 
Position = [200 0 1200 900];
set(h,'Position', Position);
savefig('Jresults/demo_region.fig');
hgload('Jresults/demo_region.fig');
setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')

% random 
vJ = reshape(J, [nrow*ncol, 1]);
class = find(vJ == background);
perm = randperm(numel(class));  %  random sampling
breakpoint = round(numel(class)*samplingRate);
trainingIndex = class(perm(1:breakpoint));
T_random = vJ;
T_random(trainingIndex) = foreground;
T_random = reshape(T_random, [nrow, ncol]);
h = figure; 
imshow(T_random); % check the training samples 
Position = [200 0 1200 900];
set(h,'Position', Position);
savefig('Jresults/demo_random.fig');
hgload('Jresults/demo_random.fig');
setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')

% filter 
filter = fspecial('gaussian', [9, 9], 2);
ST_region = imfilter(T_region, filter, 'replicate');
h = figure; imshow(ST_region);
Position = [200 0 1200 900];
set(h,'Position', Position);
hold on,
rectangle('Position',[midpoint(1)-floor(width/2), midpoint(2)- ...
      floor(height/2), width-1, height-1], 'LineWidth', 1, 'LineStyle','--');
savefig('Jresults/demo_region_filter.fig');
hgload('Jresults/demo_region_filter.fig');
setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')

ST_random = imfilter(T_random, filter, 'replicate');
h = figure; imshow(ST_random,[]);
% hold on, 
Position = [200 0 1200 900];
set(h,'Position', Position);
savefig('Jresults/demo_random_filter.fig');
hgload('Jresults/demo_random_filter.fig');
setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')