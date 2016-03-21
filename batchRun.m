function batchRun(folder, extention, myfunc)
%%batch reads hyperspectral data in the specific folder, then use myfunc to 
%%process it
% cd(folder);
list = dir(fullfile(folder,extention));
n = length(list);
for i = 1:n
    filename = list(i).name;
    myfunc(filename);
    disp(filename);
end
