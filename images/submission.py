import collections
import numpy as np

############################################################
# Problem 2

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    centroids = np.random.randn(patches.shape[0],k)

    numPatches = patches.shape[1]

    for i in range(maxIter):
        # BEGIN_SOLUTION
        # array to store distance for each patch to each centroid
        distances = np.empty((k,numPatches))

        #Step 1: Compute distances from each patch to each centroid
        for c in range(k):
            centroid = centroids[:,c].reshape(-1,1)
            d = np.sqrt(((patches-centroid)**2).sum(0))
            distances[c,:] = d

        #Step 2: Update centroids to be mean of patches in their cluster
        mins = np.argmin(distances,axis=0)
        prevCentroids = centroids.copy()
        rss = 0

        for c in range(k):
            centroids[:,c] = np.mean(patches[:,mins==c],axis=1)
            rss += np.sum(distances[c,mins==c]**2)

        # print "K-Means: RSS at iteration %d/%d is %f"%(i+1,maxIter,rss)
        # END_SOLUTION

    return centroids

############################################################
# Problem 3

def extractFeatures(patches,centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array os size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches,k))

    # BEGIN_SOLUTION
    # get distance for every patch from each centroid
    for c in range(k):
        features[:,c] = np.sqrt(((patches-centroids[:,c].reshape(-1,1))**2).sum(0))

    # threshold function (k-means triangle)
    for p in range(numPatches):
        mean_dist = np.mean(features[p,:])
        features[p,:] = np.maximum(mean_dist-features[p,:],0)

    # END_SOLUTION
    return features

############################################################
# Problem 4a

def logisticGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """
    h = 1/(1+np.exp(-theta.dot(featureVector)))
    return (h-y)*featureVector

############################################################
# Problem 4b
    
def hingeLossGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_SOLUTION
    h = theta.dot(featureVector)
    if y==0: 
        y=-1
    if (y*h)<1:
        return -float(y)*featureVector

    return np.zeros(featureVector.shape)
    # END_SOLUTION

