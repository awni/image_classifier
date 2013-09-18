'''
File: FeatureLearner
----------------
This is your file to modify! You should fill in both
the runKmeans and extractFeatures functions. 
'''

import util
import numpy as np

PATCH_LENGTH = 64

class FeatureLearner(object):

    def __init__(self, k):
        """
        Constructor
        -----------
        Called when the featureLearner is first created.
        DON"T CHANGE THIS USED FOR GRADING
        """
        self.maxIter = 10
        self.trained = False
        self.k = k
        self.centroids = None

    def runKmeans(self, trainImages):
        """
        Function: Run K-Means
        -------------
        Given a set of training images, learns self.k centroids and puts
        them in self.centroids. The function does not return a value.
        """
        assert not self.trained

        # This line starts you out with random patches in a matrix with 64
        # rows and k columns. Each col is a centroid. Each centroid has 64
        # values.
        self.centroids = np.random.randn(PATCH_LENGTH, self.k)

        ### YOUR CODE HERE ###

        # Compile all patches into one big 2-D array (patchSize by numPatches)
        patches = np.hstack([np.array(image.getPatches()).transpose() \
                                 for image in trainImages])

        numPatches = patches.shape[1]

        # array to store distance for each patch to each centroid
        distances = np.zeros((self.k,numPatches))

        for i in range(self.maxIter):

            #Step 1: Compute distances from each patch to each centroid
            for c in range(self.k):
                centroid = self.centroids[:,c].reshape(-1,1)
                d = np.sqrt(((patches-centroid)**2).sum(0))
                distances[c,:] = d
			
            #Step 2: Update centroids to be mean of patches in their cluster
            mins = np.argmin(distances,axis=0)
            prevCentroids = self.centroids.copy()
            rss = 0

            for c in range(self.k):
                # make sure something assigned to centroid
                if (mins==c).any():
                    self.centroids[:,c] = np.mean(patches[:,mins==c],axis=1)
                    rss += np.sum(distances[c,mins==c]**2)
                else:
                    # reassign to a random patch if nothing assigned to that centroid
                    print "No patches assigned to centroid %d"%c
                    self.centroids[:,c] = patches[:,int(np.random.rand()*numPatches)]
                
            print "K-Means: RSS at iteration %d/%d is %f"%(i+1,self.maxIter,rss)
        
        ### END CODE ###

        self.trained = True

    def extractFeatures(self, image):
        """
        Function: Extract Features
        -------------
        Given an image, extract and return its features. This function
        will be called many times. Should return a 1-d feature array that
        is number of patches by number of centroids long.
        """
        assert self.trained

        # populate features with features for each patch of the image.
        features = np.zeros((len(image.getPatches())*self.k))

        ### YOUR CODE HERE ###

        patches = image.getPatches()
        num_patches = len(patches)

        patches = np.array(patches).transpose()
        features = np.empty((num_patches,self.k))

        # get distance for every patch from each centroid
        for c in range(self.k):
            features[:,c] = np.sqrt(((patches-self.centroids[:,c].reshape(-1,1))**2).sum(0))

        # threshold function (k-means triangle)
        for p in range(num_patches):
            mean_dist = np.mean(features[p,:])
            features[p,:] = np.maximum(mean_dist-features[p,:],0)

        features = features.ravel()

        ### END CODE ###

        return features

