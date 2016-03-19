function feat = mm(dataCube)
% Jie Liang
% mathematical morphological methods 
% implement the extended morphological profiles method for hyperspectral image classfication 
% reference - Fauvel, M.; Tarabalka, Y.; Benediktsson, J.A.; Chanussot, J.; Tilton, J.C., "Advances in Spectral-Spatial Classification of Hyperspectral Images," Proceedings of the IEEE , vol.101, no.3, pp.652,675, March 2013

debug = false; % show figures and output variables
% step 1 Extract the majority information and reduce band numbers
dataCube = single(dataCube);
[m, n, b] = size(dataCube);
vdataCube = reshape(dataCube, [m*n, b]);

% PCA
numfPC = 3;
[~, score,~] = princomp(vdataCube); 
vdataCubePc = score(:, 1:numfPC);
dataCubePc = reshape(vdataCubePc, [m, n, numfPC]);

% step 2 Morphological process
numofSe = 4; 
dimofMp = numfPC*(numofSe*2 + 1); % dimension of morphological profile
emp = zeros(m, n, dimofMp); % EMP feature
for indexofPc = 1:numfPC
   sizeofSe = 1; %initial radius of Se not size 
   pc = dataCubePc(:,:,indexofPc);
   mid = (indexofPc-1)* (numofSe*2 + 1) + (numofSe + 1) ; 
   emp(:,:, mid) = pc ;
   for indexofSe = 1:numofSe
      se = strel('disk', sizeofSe);
     % marker = imerode(pc,se); % opening by reconstruction
      opening = imopen(pc, se);
      emp(:,:, mid + indexofSe) = opening;
     % marker = imdilate(pc, se); % closing by reconstruction
%     temp = imreconstruct(imcomplement(marker), imcomplement(pc));
      closing = imclose(pc,se);
      emp(:,:, mid - indexofSe) = closing;
      sizeofSe = sizeofSe + 2;
   end
end
emp = normalise(emp,[], 1);
if debug == true
   figure, 
   for i = 1:size(emp,3)
       slice = emp(:,:, i);
       imshow(slice, []);
       pause(1);
   end
   export2base;
end
% step 3 Feature fusion
% stack emp with original spectral feature
dataCube = normalise(dataCube, [], 1);
feat = cat(3, dataCube, emp);

function export2base
w = evalin('caller','who');
n = length(w);
for i = 1:n
    assignin('base',w{i},evalin('caller',w{i}))
end