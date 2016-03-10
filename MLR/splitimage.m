function p = splitimage(XX,w)

% x -> spectral vectors
% save classes
% y=x(2,:);

[d,n] =size(XX);
n1 = floor(n/80);
p = [];
for i = 1:79
    K = [ones(1,n1); XX(:,((i-1)*n1+1):n1*i)];   
    p1= mlogistic(w,K);
    p = [p p1];
end

clear x

K = [ones(1,n-79*n1);XX(:,(79*n1+1):n)];
        
p1=mlogistic(w,K);
p = [p p1];