% draw the overlap under controlled region with different sampling rate
clear, close all,
% nrow = 128; ncol = 152;
nrow = 65; ncol = 77;
halfrow = floor(nrow/2); halfcol = floor(ncol/2);
midpoint = [halfrow+1, halfcol+1];
J = ones(nrow,ncol);
% J = zeros(nrow,ncol);
samplingRateList = [0.05, 0.1, 0.25];
for i = 1:3
    samplingRate = samplingRateList(i);
    numofTraining = floor(nrow * ncol * samplingRate);
    temp = round(sqrt(numofTraining));
    width = round(temp/2)*2+1;
    height = width;
    indexWidth = [midpoint(1)-floor(width/2) : midpoint(1) + floor(width/2)];
    indexHight = [midpoint(2)-floor(height/2) : midpoint(2)+floor(height/2)];
    T_region = J;
    T_region(round(indexHight),round(indexWidth)) = 0;
%     T_region(indexHight,indexWidth) = 1;
    h = figure; 
    imshow(T_region); % check the training samples 
    Position = [200 0 1200 900];
    set(h,'Position', Position);
    figname = ['Jresults/', 'demo_region', num2str(samplingRate*100), '.fig'];
    savefig(figname);
    hgload(figname);
    setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')
    
    % filter 
    filter = fspecial('gaussian', [15, 15], 5);
%     filter = fspecial('average', [21, 21]);
    ST_region = imfilter(T_region, filter, 'replicate');

    h = figure; 
    imshow(ST_region);
    Position = [200 0 1200 900];
    set(gcf,'Position', Position);
    hold on,
    rectangle('Position',[midpoint(1)-floor(width/2), midpoint(2)- ...
        floor(height/2), width-1, height-1], 'LineWidth', 2, 'LineStyle','--');
    figname = ['Jresults/', 'demo_region', num2str(samplingRate*100), '_filter.fig'];
    savefig(figname);
    hgload(figname);
    setImage('C:\Users\s2882161\Google Drive\working\TGRS2015\respons_letter\images')
end