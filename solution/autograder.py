

### Autograder.py ###
# Runs 7 tests comparing student generated
# results to ground truth results.
# Scores students out of 60 points.


import numpy as np
import util
import pickle
import sys

# Test Kmeans - Similar to test in evaluator.py
def test_kmeans(stress=False):
    import studentLearner as fl

    if stress:
        np.random.seed(133)
        k = 10
        learner = fl.FeatureLearner(k)
        learner.maxIter = 20
        testfile = 'grading_data/kmeans_stress.npy'
        tr = util.loadTrainImages()[100:200]
    else:
        np.random.seed(142)
        k = 5
        learner = fl.FeatureLearner(k)
        learner.maxIter = 10
        testfile = 'grading_data/kmeans_simple.npy'
        tr = util.loadTrainImages()[:100]

    # load test centroids
    testDat = open(testfile,'r')
    centroids = pickle.load(testDat)
    testDat.close()

    learner.runKmeans(tr)

    # check the basics
    if not isinstance(learner.centroids,np.ndarray):
        return False,"Centroids should be stored in numpy array"
    if len(learner.centroids.shape) != 2:
        return False,"centroids array should be 2D"
    if learner.centroids.shape[0] != util.patch_dim**2:
        return False,"Size of centroids not correct"
    if learner.centroids.shape[1] != k:
        return False,"Number of centroids not correct"
    if not np.all(~np.isnan(learner.centroids)):
        return False,"NaNs detected in centroids"
    # check that they are the same
    diff = np.sum((centroids.reshape([-1])-learner.centroids.reshape([-1]))**2)

    if diff > 1e-5:
        return False,"Centroid mismatch"
    
    return True,"Passed!"


# Test Feature Extraction - Uses solution centroids.
def test_feature_extraction(stress=False):

    import studentLearner as fl
    if stress:
        np.random.seed(133)
        k = 10
        testCentroidFile='grading_data/kmeans_stress.npy'
        testFeaturesFile='grading_data/features_stress.npy'
        images = util.loadTrainImages()[500:600]
    else:
        np.random.seed(142)
        k = 5
        testCentroidFile='grading_data/kmeans_simple.npy'
        testFeaturesFile='grading_data/features_simple.npy'
        images = [util.loadTrainImages()[99]]

    learner = fl.FeatureLearner(k)
    learner.trained = True

    # load test centroids and features
    testDat = open(testCentroidFile,'r')
    centroids = pickle.load(testDat)
    testDat.close()
    testDat = open(testFeaturesFile,'r')
    features = pickle.load(testDat)
    testDat.close()

    learner.centroids = centroids

    for i in range(len(images)):
        studentFeats = learner.extractFeatures(images[i]).squeeze()
        if not isinstance(studentFeats,np.ndarray):
            return False,"Features should be in a numpy array"
        if studentFeats.shape!=features[:,i].squeeze().shape:
            return False,"Dimension mismatch"

        if np.abs(np.sqrt(np.sum(features**2)) - np.sqrt(np.sum(features**2))) > 1e-3:
            return False,"Feature mismatch"


    return True,"Passed!"

# Test Logistic Regression - Similar to test in
# evaluator.py
def test_log_regression(stress=False):

    import classifier
    clf = classifier.Classifier()

    if stress:
        np.random.seed(133)
        k=10
        images = util.loadTrainImages()[100:200]
        clf.maxIter = 100
        thetaFile = 'grading_data/log_reg_stress.npy'
    else:
        np.random.seed(142)
        k = 5
        images = util.loadTrainImages()[:100]
        clf.maxIter = 5
        thetaFile = 'grading_data/log_reg_simple.npy'

    clf.train(images,k)
    studentTheta = clf.theta.squeeze()
            
    fid = open(thetaFile,'r')
    theta = pickle.load(fid)
    fid.close()

    if studentTheta.shape!=theta.shape:
        return False,"Dimension mismatch"

    if np.abs(np.sqrt(np.sum(theta**2)) -
              np.sqrt(np.sum(studentTheta**2))) > 1e-3:
        return False,"Parameter vector mismatch"

    return True,"Passed!"

# Test predictions - Tests the test method in classifier.py using ground
# truth theta
def test_predictions(dummy=True):
    np.random.seed(133)
    import classifier
    clf = classifier.Classifier()
    images = util.loadTrainImages()[100:200]
    clf.maxIter = 1
    k = 10
    clf.train(images,k)

    predfile = 'grading_data/predictions.npy'
    fid = open('grading_data/log_reg_stress.npy','r')
    theta = pickle.load(fid)
    fid.close()

    clf.theta = theta
    studentPreds = clf.test(images)
    
    fid = open(predfile,'r')
    preds = pickle.load(fid)
    fid.close()
    
    preds = np.array(preds,dtype=np.int32)
    studentPreds = np.array(studentPreds,dtype=np.int32)

    if studentPreds.size!=preds.size:
        return False,"Wrong number of predictions."

    if np.sum(preds==studentPreds)!=preds.size:
        # try again with intercept at end just in case
        fid = open('grading_data/log_reg_stress_int_end.npy','r')
        theta = pickle.load(fid)
        fid.close()
        clf.theta = theta
        studentPreds = np.array(clf.test(images),dtype=np.int32)
        fid = open('grading_data/predictions_int_end.npy','r')
        preds = np.array(pickle.load(fid),dtype=np.int32)
        fid.close()
        if np.sum(preds==studentPreds)==preds.size:
            return True,"Passed!"

        return False,"Prediction mismatch."

    return True,"Passed!"

def parseGrading():
    fid = open("GRADING",'r')
    lines = fid.readlines()
    fid.close()
    log = []
    for l in lines:
        if "Comments:" in l:
            continue
        log.append(l)
    return log

def writeFile(log,grade,outfile=".GRADING"):
    outfile = ".GRADING"
    out = open(outfile,'w')
    for line in log:
        if "Grade:" in line:
            line = "Grade:  %d/%d\n"%(grade,60)
        out.write(line)

    out.close()


def runTest(log,testname,desc,testnum,testFunc,stress,val):
    log.append("\n%s\n"%testname)
    log.append("-----------------\n")
    log.append(desc)
    grade = 0
    try:
        passed,message = testFunc(stress)
        if passed:
            grade = val
        log.append(message+"\n")
    except:
        print >> sys.stderr, "CRASHED THE AUTOGRADER!"
        print >> sys.stderr, "Test %d: %s"%(testnum,testname)
        log.append("CRASHED!\n")
        pass

    log.append("Test %d: %d/%d\n"%(testnum,grade,val))
    return grade

def main():

    grade = 0
    log = parseGrading()

    desc1 = "K-means with a few centroids on a\n"\
        "small dataset. Checks centroids are\n"\
        "within a small error of the correct\n"\
        "result.\n"
    desc2 = "K-means with more centroids on a\n"\
        "larger dataset for 20 iterations. Checks\n"\
        "centroids are within a small error of\n"\
        "correct result.\n"
    desc3 = "Feature extraction for 1 image using\n"\
        "few centroids. Checks that features are\n"\
        "within a small error of correct result.\n"\
        "Uses solution centroids.\n"
    desc4 = "Feature extraction for 100 images using\n"\
        "larger number of centroids. Checks that\n"\
        "features are within a small error of\n"\
        "correct result. Uses solution centroids.\n"
    desc5 = "Logistic regression with few centroids\n"\
        "on a small dataset for 5 iterations. Checks\n"\
        "that parameters are within a small error of\n"\
        "correct result. Uses solution features.\n"
    desc6 = "Logistic regression with more features\n"\
        "on a larger dataset for 100 iterations. Checks\n"\
        "that parameters are within a small error of\n"\
        "correct result. Uses solution features.\n"
    desc7 = "Predictions on 100 images.  Compares to\n"\
        "correct predictions and passes if they are\n"\
        "the same. Uses solution parameters.\n"

    tests = [["K-means Simple",desc1,test_kmeans,False,15],["K-means Stress",desc2,test_kmeans,True,5],["Feature Extraction Simple",desc3,test_feature_extraction,False,10],["Feature Extraction Stress",desc4,test_feature_extraction,True,5],["Logistic Regression Simple",desc5,test_log_regression,False,15],["Logistic Regression Stress",desc6,test_log_regression,True,5],["Classifier Predictions",desc7,test_predictions,True,5]]

    testNum = 0
    for test in tests:
        testNum += 1
        grade += runTest(log,test[0],test[1],testNum,test[2],test[3],test[4])

    writeFile(log,grade,".GRADING")

if __name__=='__main__':
    main()




