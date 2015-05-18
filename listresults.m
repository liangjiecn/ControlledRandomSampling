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
for index = 1:numdataset
   filename = datalist(index).name;
   indexof_= find(filename == '_',1);
   dataset{index} = filename(1:indexof_-1);
   results = importdata(filename);
   p5(index) = getfield(results,{1}, fieldname);
   p10(index) = getfield(results,{2}, fieldname);
   p25(index) = getfield(results,{3}, fieldname);
%    AA = results.AA;
%   kappa(index) = results.kappa;
end 
T = table(p5, p10, p25, 'RowNames',dataset)
cd .. 