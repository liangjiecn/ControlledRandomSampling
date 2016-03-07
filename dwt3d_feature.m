% Syntax:
% feats=dwt3d_feature(DataCube) ;
% where DataCube is the 3D hyperspectral data cube with size m*n*p, and the
% output is the feature cube with size  m*n*15p, which is stacked with 15 
% wavelet sub-band features, i.e., when the input hyperspectral data has p 
% bands, the output feature size is 15p.
%
% *Note: the features have been already normalized, so do NOT perform any
% other normalization step.
% % modified by Jie Liang to reduce momory
function feats=dwt3d_feature(DataCube) 
DataCube=double(DataCube);
[m, n, b] = size(DataCube);
feats=dwt3_2(DataCube,'haar');
feats=feats.dec;
d1=dwt3_2(feats{1,1,1},'haar');
d1=d1.dec;
d1=cat(3,d1{:});
d1 = single(d1); % LJ
feats=cat(3,feats{:});
feats=feats(:,:,size(DataCube,3)+1:end);
feats = single(feats); % LJ
feats=cat(3,d1,feats);
clear d1 DataCube;
feats=abs(feats);
filter_mask=1/9*ones(3,3);
for i=1:size(feats,3)
    feats(:,:,i)=conv2( feats(:,:,i),filter_mask,'same');
end
feats=reshape(feats,[m*n, b*15]);
feats1 = feats(1:round(end/2),:); % LJ to reduce computational burden
feats2 = feats(round(end/2)+1:end,:); % LJ
clear feats; %LJ
feats1 = mapstd(feats1);  % LJ
feats2 = mapstd(feats2); % LJ
feats = cat(1,feats1, feats2);
feats=reshape(feats,[m,n,b*15]);



% 3D DWT without down-sampling
function wt = dwt3_2(X,varargin)

% Check arguments.
nbIn = nargin;
msg = nargchk(2,4,nbIn);
if ~isempty(msg)
    error('Wavelet:FunctionInput:NbArg',msg)
end
LoD = cell(1,3); HiD = cell(1,3); LoR = cell(1,3); HiR = cell(1,3);
argStatus = true;

if ischar(varargin{1})
    [LD,HD,LR,HR] = wfilters(varargin{1});
    for k = 1:3
        LoD{k} = LD; HiD{k} = HD; LoR{k} = LR; HiR{k} = HR;
    end
    
elseif isstruct(varargin{1})
    if isfield(varargin{1},'w1') && isfield(varargin{1},'w2') && ...
            isfield(varargin{1},'w3')
        for k = 1:3
            [LoD{k},HiD{k},LoR{k},HiR{k}] = ...
                wfilters(varargin{1}.(['w' int2str(k)]));
        end
    elseif isfield(varargin{1},'LoD') && isfield(varargin{1},'HiD') && ...
            isfield(varargin{1},'LoR') && isfield(varargin{1},'HiR')
        for k = 1:3
            LoD{k} = varargin{1}.LoD{k}; HiD{k} = varargin{1}.HiD{k};
            LoR{k} = varargin{1}.LoR{k}; HiR{k} = varargin{1}.HiR{k};
        end
    else
        argStatus = false;
    end
    
elseif iscell(varargin{1})
    if ischar(varargin{1}{1})
        for k = 1:3
            [LoD{k},HiD{k},LoR{k},HiR{k}] = wfilters(varargin{1}{k});
        end
    elseif iscell(varargin{1})
        Sarg = size(varargin{1});
        if isequal(Sarg,[1 4])
            if ~iscell(varargin{1}{1})
                LoD(1:end) = varargin{1}(1); HiD(1:end) = varargin{1}(2);
                LoR(1:end) = varargin{1}(3); HiR(1:end) = varargin{1}(4);
            else
                LoD = varargin{1}{1}; HiD = varargin{1}{2};
                LoR = varargin{1}{3}; HiR = varargin{1}{4};
            end
        elseif isequal(Sarg,[3 4])
            LoD = varargin{1}(:,1)'; HiD = varargin{1}(:,2)';
            LoR = varargin{1}(:,3)'; HiR = varargin{1}(:,4)';
        else
            argStatus = false;
        end
    end
else
    argStatus = false;
end
if ~argStatus
    error('Wavelet:FunctionInput:ArgVal','Invalid argument value!');
end
sX = size(X);

X = double(X);
dec = cell(2,2,2);
permVect = [];
[a_Lo,d_Hi] = wdec1D(X,LoD{1},HiD{1},permVect);
permVect = [2,1,3];
[aa_Lo_Lo,da_Lo_Hi] = wdec1D(a_Lo,LoD{2},HiD{2},permVect);
[ad_Hi_Lo,dd_Hi_Hi] = wdec1D(d_Hi,LoD{2},HiD{2},permVect);
permVect = [1,3,2];
[dec{1,1,1},dec{1,1,2}] = wdec1D(aa_Lo_Lo,LoD{3},HiD{3},permVect);
[dec{2,1,1},dec{2,1,2}] = wdec1D(da_Lo_Hi,LoD{3},HiD{3},permVect);
[dec{1,2,1},dec{1,2,2}] = wdec1D(ad_Hi_Lo,LoD{3},HiD{3},permVect);
[dec{2,2,1},dec{2,2,2}] = wdec1D(dd_Hi_Hi,LoD{3},HiD{3},permVect);

wt.sizeINI = sX;
wt.filters.LoD = LoD;
wt.filters.HiD = HiD;
wt.filters.LoR = LoR;
wt.filters.HiR = HiR;
wt.mode = 'sym';
wt.dec = dec;



%-----------------------------------------------------------------------%
function [L,H] = wdec1D(X,Lo,Hi,perm)

if ~isempty(perm) , X = permute(X,perm); end
sX = size(X);
if length(sX)<3 , sX(3) = 1; end

lf = length(Lo);
lx = sX(2);
if lx<lf+1
    nbAdd = lf-lx+1;
    Add = zeros(sX(1),nbAdd,sX(3));
    X = [Add , X , Add];
end

X = [X(:,lf-1:-1:1,:) , X , X(:,end:-1:end-lf+2,:)];

L = convn(X,Lo,'same');
H = convn(X,Hi,'same');
clear X;
lenL = size(L,2);
first = lf; last = lenL-lf+1;
L = L(:,first:last,:); H = H(:,first:last,:);

if ~isempty(perm)
    L = permute(L,perm);
    H = permute(H,perm);
end
%-----------------------------------------------------------------------%
