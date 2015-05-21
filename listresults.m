function T = listresults(feature, method, sampling, varargin )
% put results in table
resultsPath = 'Jresults';
cd (resultsPath);
if isempty(varargin)
   fieldname = 'OA'; 
else
   fieldname = varargin{1}; 
end
context = ['*_',method,'*',feature,'*',sampling,'*'];
datalist =  dir(context);
numdataset = length(datalist);
dataset = cell(numdataset, 1);
p5 = zeros(numdataset, 1);
p10 = zeros(numdataset, 1);
p25 = zeros(numdataset, 1);
p5std = zeros(numdataset, 1);
p10std = zeros(numdataset, 1);
p25std = zeros(numdataset, 1);
for index = 1:numdataset
   filename = datalist(index).name;
   indexof_= find(filename == '_',1);
   dataset{index} = filename(1:indexof_-1);
   results = importdata(filename);
   [numr, numc] = size(results);
   tempval = zeros(numr, numc);
   for i = 1: numr
       for j = 1: numc
           ind = sub2ind([numr, numc], i, j);
           tempval(i,j) = getfield(results,{ind}, fieldname); 
       end
   end
   p5(index) = mean(tempval(1,:));
   p5std(index) = std(tempval(1,:));
   p10(index) = mean(tempval(2,:));
   p10std(index) = std(tempval(2,:));
   p25(index) = mean(tempval(3,:));
   p25std(index) = std(tempval(3,:));
%    AA = results.AA;
%   kappa(index) = results.kappa;
end 
T = table(p5, p5std, p10, p10std, p25, p25std, 'RowNames',dataset)
cd .. 