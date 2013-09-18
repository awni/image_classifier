function [patches, whiten] = extract_patches(images,im_x,im_y,rf)


m=size(images,2);
n=size(images,1);

%% extract patches
numpatches = (im_x/rf)^2*m;
patches = zeros(rf.^2,numpatches);
p=1;
for i=1:m
    
    % grab patches from each image
    image = images(:,i);
    image = reshape(image,im_x,im_y);
    
    for x=1:rf:im_x
        for y=1:rf:im_y
            patch = image(x:x+rf-1,y:y+rf-1);
            patches(:,p) = patch(:);
            p = p+1;
        end;
    end;
    
end;


%% Remove DC component and contrast normalize patches
mean_patches = mean(patches);
var_patches = var(patches);
patches = bsxfun(@minus,patches,mean_patches);
patches = bsxfun(@rdivide,patches,sqrt(var_patches+10));

%% ZCA Whiten
Sigma = cov(patches');
[U V] = svd(Sigma);
eps = 0.01;
whiten = U*diag((diag(V)+eps).^(-1/2))*U';

patches = whiten*patches;







%% Predictions






