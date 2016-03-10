function p = mlogistic(w,x)

% compute the  multinomial distributions (one per sample)

%   Authors:  Jose Bioucas-Dias, 2007
m = size(w,2)+1;

aux = exp(w'*x);
p =  aux./repmat(1+sum(aux,1),m-1,1);

% last class
p(m,:) = 1-sum(p,1);