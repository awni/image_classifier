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

    def __init__(self):
        """
        Constructor
        -----------
        Called when the classifier is first created.
        DONT CHANGE THIS USED FOR GRADING
        """
        self.trained = False
        self.alpha = 1e-5
        self.maxIter = 5000
        self.featureLearner = None
        self.theta = None

    def train(self, trainImages, k):
        """
        Function: Train
        -------------
        Given a set of training images, and a number of centroids to learn,
        k, calculate any information you will need in order to make
        predictions in the testing phase.
        """
        assert not self.trained

        # We are going to use the Feature Learner you programmed
        # in the first two parts of the assignment.
        self.featureLearner = FeatureLearner(k)

        # First run your K-means function. After, you will be
        # able to use the extractFeatures method that you wrote.
        self.featureLearner.runKmeans(trainImages)
        
        # Initialize weight vector
        numPatches = (util.image_x/util.patch_dim)**2
        self.theta = 1e-2*np.random.randn(k*numPatches+1)

        ### YOUR CODE HERE ###

        # Extract features
        X = [np.array(self.featureLearner.extractFeatures(image)) for image in
        trainImages]
        
        X = np.vstack(X).transpose() # featdim by num samples
        X = np.vstack([np.ones(X.shape[1]), X])

        # label array
        Y = np.array([image.getLabel() for image in trainImages])

        # run gradient descent on learned features
        for i in range(self.maxIter):
            
            # logistic probabilities
            probs = 1/(1+np.exp(-self.theta.dot(X)))

            cost = -(np.sum(np.log(probs[Y==1]))+ \
                np.sum(np.log(1-probs[Y==0])))

            if i%100 == 0:
                print "Logistic Regression: Cost on iteration %d is %f"%(i,cost)


            grad = np.sum((probs-Y)*X,axis=1) # calculates gradient
            self.theta = self.theta - self.alpha*grad
            
        ### END CODE ###

        self.trained = True

    def test(self, testImages):
        """
        Function: Test
        -------------
        Given a set of testing images make a prediction for each
        image. You may assume that the train function has already been
        called.
        """
        assert self.trained
        
        # populate this list with best guess for each image
        predictions = []

        ### YOUR CODE HERE ###

        # Extract features
        X = [self.featureLearner.extractFeatures(image) for image in
        testImages]
        
        X = np.vstack(X).transpose() # featdim by num samples
        X = np.vstack([np.ones(X.shape[1]),X])

        # probabilities
        probs = 1/(1+np.exp(-self.theta.dot(X)))

        predictions = np.zeros(probs.shape)
        predictions[probs>0.5] = 1
        predictions = predictions.tolist()

        ### END CODE ###

        return predictions
