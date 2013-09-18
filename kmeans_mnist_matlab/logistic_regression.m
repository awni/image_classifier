function theta = logistic_regression(X,Y)

%% Supervised training

n = size(X,1);
m = size(X,2);

theta = randn(n,1); % parameter vec

alpha = 0.1;
% minimize negative likelihood for binomial logistic
% run gradient descent
maxIter = 3000;
for it=1:maxIter
    h = (1./(1+exp(-theta'*X)));
    
    % cost = -(1/m)*sum(Y.*log(h)+(1-Y).*log(1-h));
    
    % avoids NaNs
    cost = -(1/m)*(sum(log(h(Y==1)))+sum(log(1-h(Y==0))));
    
    grad = (1/m)*sum(bsxfun(@times,X,h-Y),2);
    theta = theta - alpha*grad;
    if mod(it,100)==0
        fprintf('Iteration %d/%d: Negative Log Likelihood is %f\n',it,maxIter,cost);
    end;
    
end;