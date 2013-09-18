%% Separate out from MNIST ones and zeros
clear all;
addpath mnistHelper;
addpath data;

randn('seed',3);
rand('seed',7);

train_images=loadMNISTImages('data/train-images-idx3-ubyte');
train_labels=loadMNISTLabels('data/train-labels-idx1-ubyte');

%% keep only zeros and ones
keep = (train_labels==1)|(train_labels==0);
train_images = train_images(:,keep);
train_labels = train_labels(keep);

% decrease dataset size (too big)
train_images = train_images(:,1:5000);
train_labels = train_labels(1:5000);

%% Random split data for supervised and unsupervised training
keep = rand(size(train_images,2),1);
keep = keep>0.5;
data_us = train_images(:,keep);
data_s = train_images(:,~keep);
labels_s = train_labels(~keep);

%% Random subdivide supervised set into train and test
keep = rand(size(data_s,2),1);
keep = keep>0.2;
data_train_s = data_s(:,keep);
labels_train_s = labels_s(keep);
data_test_s = data_s(:,~keep);
labels_test_s = labels_s(~keep);

%% Train Unsupervised

% Extract patches
im_x = 28;
im_y = 28;
rf = 7; % should divide im_x and im_y
[patches, whiten] = extract_patches(data_us,im_x,im_y,rf);

% K-Means
K=10;
D = run_kmeans(patches,K);

% save('centroids','D');
% visualize a centroid
% imagesc(reshape(D(:,1),7,7); colormap gray;

%% Train Supervised
% Extract Features

feats = make_features(D,data_train_s,whiten,im_x,im_y);

% Logistic Regression

theta = logistic_regression(feats,labels_train_s');

%% Test

% Extract Features

test_feats = make_features(D,data_test_s,whiten,im_x,im_y);

% Make predictions

probs = 1./(1+exp(-theta'*test_feats));
predictions = probs>0.5;

accuracy = sum(predictions==labels_test_s')/length(labels_test_s);
fprintf('Accuracy is %2.2f%%\n',accuracy*100);




