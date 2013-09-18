function feats = make_features(D,X,whiten,im_x,im_y)
%% Extract features using learned centroids

m=size(X,2);
rf = sqrt(size(D,1));
K = size(D,2);

feats = zeros(m,K,im_x/rf,im_y/rf);

for i=1:m
    
    image = X(:,i);
    image = reshape(image,im_x,im_y);
    
    for x=1:rf:im_x
        for y=1:rf:im_y
            
            % extract patch and whiten
            patch = image(x:x+rf-1,y:y+rf-1);
            patch = patch(:);
            patch = patch-mean(patch);
            patch = patch/sqrt(var(patch)+10);
            patch = whiten*patch;
            
            % extract features
            indx = (x+rf-1)/rf;
            indy = (y+rf-1)/rf;
            for k=1:K
                feats(i,k,indx,indy) = norm(patch-D(:,k));
            end;
            
            % take only above average features
            feats(i,:,indx,indy) = max(mean(feats(i,:,indx,indy))-...
                    feats(i,:,indx,indy),0);
            
        end;
    end;
    
end;

% unroll feats into n x m matrix

feats = reshape(feats,m,[]);
feats = feats';