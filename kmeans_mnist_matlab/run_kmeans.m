function D = run_kmeans(patches,K)

%% Runs K-means
rf = size(patches,1); % size of receptive field
m = size(patches,2); % num patches


D = randn(rf,K); % random initialize centroids from gaussian
D = bsxfun(@rdivide,D,sum(D,1)); % normalize

% update centroids
maxIter=50;
distances = zeros(K,m);
for it=1:maxIter
    Dprev = D;
    for k=1:K
        distances(k,:)=norms(bsxfun(@minus,patches,D(:,k)),2);
    end;
    [~,x]=min(distances);
    
    for k=1:K
       % check if we are getting zero samples for centroid
       if sum(x==k)~=0
           D(:,k) = sum(patches(:,x==k),2)./sum(x==k);
       else
           fprintf('Got zero on centroid %d\n',k)
           D(:,k) = patches(:,ceil(rand(1)*m));
       end;
    end;
    diffs=Dprev-D;
    maxdiff=max(diffs(:));
    fprintf('K-Means Iteration %d/%d: max diff is %f\n',it,maxIter,maxdiff);
end;


