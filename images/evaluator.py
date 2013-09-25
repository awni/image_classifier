'''
File: Evaluator
---------------
Runs and evaluates your Classifier. You should *NOT* modify this
file. Instead you will program featureLearner.py and classifier.py
'''
from classifier import Classifier
import util
import pickle
import  numpy as np

class Evaluator(object):

    def run(self):
        """
        Function: Run
        -------------
        This function will evaluate your solution! You do not need to
        write any code in this file, however you SHOULD understand this
        function!
        """
        print "Running the full pipeline!"
        K=25
        trainImages = util.loadTrainImages()[:1000]
        testImages = util.loadTestImages()

        classifier = Classifier()

        print 'Training..........'
        classifier.train(trainImages, K)

        trainPredictions = classifier.test(trainImages)
        trainAccuracy = self.evaluate(trainPredictions, trainImages)

        print 'Testing...........'
        testPredictions = classifier.test(testImages)
        testAccuracy = self.evaluate(testPredictions, testImages)

        print 'All done. Here is your summary:'
        self.reportAccuracy(trainAccuracy, 'Train Accuracy')
        self.reportAccuracy(testAccuracy, 'Test Accuracy')

    def evaluate(self, predictions, images):
        """
        Function: Evaluate
        ------------------
        Checks how well your program did at classifying images.
        """
        if len(predictions) != len(images):
            raise Exception('Your prediction list is not the same length as the image list!')
        numCorrect = 0
        for i in range(len(images)):
            image = images[i]
            prediction = predictions[i]
            if prediction == image.label:
                numCorrect += 1
        accuracy = float(numCorrect) / len(images)
        return accuracy
        
    def reportAccuracy(self, accuracy, label):
        """
        Function: Report Accuracy
        ------------------
        Prints accuracy result to the screen.
        """
        print '-------------------------'
        print label + ': ' + str(accuracy)

    def runDev(self):
        """
        Function: runDev
        -------------
        This function will run the full pipeline in development mode.
        I.e. it will use only 10 centroids and 100 images.
        """
        print "Running in development mode"

        K=5
        trainImages = util.loadTrainImages()[:100]
        testImages = util.loadTestImages()[:100]
        
        classifier = Classifier()
        
        print 'Training..........'
        classifier.train(trainImages, K)
        trainPredictions = classifier.test(trainImages)
        trainAccuracy = self.evaluate(trainPredictions, trainImages)

        print 'All done. Here is your summary:'
        self.reportAccuracy(trainAccuracy, 'Train Accuracy')

###############################################
# Students don't need to read any code beyond #
# this point!!                                #
###############################################

# test harness for student k-means implementation
def test_kmeans():
    np.random.seed(33)
    print "Testing K-Means implementation..."

    import featureLearner as fl
    k = 5
    learner = fl.FeatureLearner(k)
    learner.maxIter = 10

    tr = util.loadTrainImages()[:100]
    learner.runKmeans(tr)
    
    # check the basics
    assert isinstance(learner.centroids,np.ndarray),"centroids should be stored in numpy array"
    assert len(learner.centroids.shape) == 2, "centroids array should be 2-D"
    assert learner.centroids.shape[0] == util.patch_dim**2, "Size of centroids not correct"
    assert learner.centroids.shape[1] == k,"Number of centroids not correct"

    # load test centroids
    testDat = open('data/kmeans_test.npy','r')
    centroids = pickle.load(testDat)
    testDat.close()

    # check that they are the same
    diff = np.sum((centroids.reshape([-1])-learner.centroids.reshape([-1]))**2)

    if diff > 1e-5:
        print "Somethings wrong, your centroids don't match the test centroids"
    else:
        print "K-means test passed"

# test harness for feature extraction implementation
def test_feature_extraction():
    np.random.seed(33)
    print "Testing implementation of feature extraction..."

    import featureLearner as fl
    k = 5
    learner = fl.FeatureLearner(k)
    learner.trained = True

    image = util.loadTrainImages()[33]

    # load test centroids and features
    testDat = open('data/kmeans_test.npy','r')
    centroids = pickle.load(testDat)
    testDat.close()
    testDat = open('data/features_test.npy','r')
    features = pickle.load(testDat)
    testDat.close()

    learner.centroids = centroids

    studentFeats = learner.extractFeatures(image).squeeze()
    
    assert isinstance(studentFeats,np.ndarray),"Features should be in a numpy array"
    assert studentFeats.shape==features.shape,"Dimension mismatch"
    
    if np.abs(np.sum(features)-np.sum(studentFeats)) > 1e-3:
        print "Feature mismatch, test failed"
        return
    
    if np.abs(np.sqrt(np.sum(features**2)) - np.sqrt(np.sum(features**2))) > 1e-3:
        print "Feature mismatch, test failed"
        return

    print "Feature extraction test passed"    
        

        
# test harness for logistic regression implementation
def test_log_regression():
    np.random.seed(33)
    print "Testing implementation of logistic regression..."

    import classifier
    k = 5
    images = util.loadTrainImages()[:100]

    clf = classifier.Classifier()
    clf.maxIter = 5
    clf.train(images,k)

    fid = open('data/log_reg_test.npy','r')
    theta = pickle.load(fid)
    fid.close()

    studentTheta = clf.theta.squeeze()
    
    assert studentTheta.shape==theta.shape,"Dimension mismatch"

    if np.abs(np.sum(theta)-np.sum(studentTheta)) > 1e-3:
        print "Parameter vector mismatch, test failed"
        return

    if np.abs(np.sqrt(np.sum(theta**2)) - np.sqrt(np.sum(studentTheta**2))) > 1e-3:
        print "Parameter vector mismatch, test failed"
        return

    print "Logistic regression test passed."

def kmeans_only(view=False):
    print "Running only  K-Means..."
    import featureLearner as fl
    k = 25
    learner = fl.FeatureLearner(k)

    tr = util.loadTrainImages()[:1000]
    learner.runKmeans(tr)

    if view:
        util.viewPatches(learner.centroids)

def main(args=None):
    """
    Function: Main
    ------------------
    Creates and runs the Evaluator
    """
    from optparse import OptionParser
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-d","--dev", dest="dev_mode",action="store_true",
                      default=False,help="Run pipeline with small constants for development")
    parser.add_option("-f","--seed", dest="seed",action="store_true",
                      default=False,help="Seed random number generator")
    parser.add_option("-k","--kmeans",dest="test_kmeans",action="store_true",
                      default=False,help="Run only K-Means")
    parser.add_option("-e","--extract",dest="test_extract",action="store_true",
                      default=False,help="Test implementation of Feature Extraction")
    parser.add_option("-s","--supervised",dest="test_supervised",action="store_true",
                      default=False,help="Test implementation of Logistic Regression")
    parser.add_option("-t","--test",dest="test",action="store_true",default=False,
                      help="tests k-means implementation")
    parser.add_option("-v","--view",dest="view",action="store_true",default=False,
                      help="pass in the -v flag along with the -k flag to view learned centroids")

    (opts,args) = parser.parse_args(args)

    if opts.seed:
        print "Setting random seed"
        np.random.seed(33)

    if opts.test_kmeans:
        if opts.test:            
            test_kmeans()
        else:
            kmeans_only(opts.view)
    elif opts.test_extract:
        test_feature_extraction()
    elif opts.test_supervised:
        test_log_regression()
    elif opts.dev_mode:
        Evaluator().runDev()
    else:
        Evaluator().run()


if __name__ == '__main__':
    main()
