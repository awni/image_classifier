'''
File: Classifier
----------------
This is your file to modify! You should fill in both
the learn and extractFeatures functions. 
'''

import util
import numpy as np

class FeatureLearner(object):

    # Constructor
    # -----------
    # Called when the classifier is first created.
    def __init__(self, k):
        #DON"T CHANGE THIS USED FOR GRADING
        self.maxIter = 50
        self.trained = False
        self.k = k
        self.centroids = None

    # Function: Learn Features
    # -------------
    # Given a set of training images, and a number of features
    # to learn k, calculate any information you will need in 
    # order to make extract a feature set of size k. This 
    # function will be called only once. 
    def runKmeans(self, trainImages):
        assert not self.trained

        # self.centroids = np.zeros(util.patch_dim**2,self.k)

        ### YOUR CODE HERE ###

        # make sure to fill in self.centroids with your learned
        # centroids

        self.trained = True

    # Function: Extract Features
    # -------------
    # Given an image, extract and return its features. This
    # function will be called many times. Should return a 1-d
    # feature array that is number of patches by number of
    # centroids long
    def extractFeatures(self, image):
        assert self.trained

        # populate features with features for each patch
        # of the image 
        features = np.empty((len(image.getPatches)*self.k))

        ### YOUR CODE HERE ###

        return features
