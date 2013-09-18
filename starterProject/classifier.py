'''
File: Classifier
----------------
This is your file to modify! You should fill in both
the train and test functions.
'''
import util
import numpy as np
from featureLearner import FeatureLearner

class Classifier(object):

    # Constructor
    # -----------
    # Called when the classifier is first created.
    def __init__(self):
        # DONT CHANGE THIS USED FOR GRADING
        self.trained = False
        self.alpha = 1e-5 # learning rate
        self.maxIter = 5000 # max num iterations
        self.featureLearner = None
        self.theta = None # parameter vector for logistic regression

    # Function: Train
    # -------------
    # Given a set of training images, and a number of centroids
    # to learn k,
    # calculate any information you will need in order to make 
    # predictions in the testing phase. This function will be
    # called only once. Your training must feature select!
    def train(self, trainImages, k):
        assert not self.trained
        self.featureLearner = FeatureLearner(k)
        
        ### YOUR CODE HERE ###

        self.trained = True

    # Function: Test
    # -------------
    # Given a set of testing images
    # calculate a list of predictions for those images. You
    # may assume that the train function has already been called. 
    # This function will be called multiple times.
    def test(self, testImages):
        assert self.trained

        # populate this list with best guess for each image
        predictions = [] 

        ### YOUR CODE HERE ###

        return predictions
