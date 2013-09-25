
import submission
import util
import numpy as np
import pickle

def run(grad,view,pixels,maxIter,numTrain):
    k = 25
    maxIter_kMeans = 20

    trainUImages = util.loadTrainImages()[:numTrain]
    trainSImages = trainUImages[:500]
    testImages = util.loadTestImages()

    if pixels is False:
        # Compile all patches into one big 2-D array (patchSize x numPatches)
        patches = np.hstack([np.array(image.getPatches()).transpose() for image in trainUImages])
        print "Training K-means using %d images"%numTrain
        centroids = submission.runKMeans(k,patches,maxIter_kMeans)
        trainX,trainY = util.kMeansFeatures(trainSImages,centroids,submission.extractFeatures)
        testX, testY = util.kMeansFeatures(testImages,centroids,submission.extractFeatures)
        if view:
            util.viewPatches(centroids)
    else:
        maxIter = 100
        trainX,trainY = util.pixelFeatures(trainSImages)
        testX,testY = util.pixelFeatures(testImages)

    clf = util.Classifier(maxIter=maxIter,alpha=5e-5,gradient=grad)
    clf.train(trainX,trainY)

    predictions = clf.test(trainX)
    acc = np.sum(trainY==predictions)/float(trainY.size)
    print "Train accuracy is %f"%acc

    predictions = clf.test(testX)
    acc = np.sum(testY==predictions)/float(testY.size)
    print "Test accuracy is %f"%acc

def main(args=None):

    from optparse import OptionParser
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-f","--seed", dest="seed",action="store_true",
                      default=False,help="Seed random number generator")
    parser.add_option("-g","--gradient",dest="grad",
                      default="perceptron",help="Gradient of loss function for weight update")
    parser.add_option("-v","--view",dest="view",action="store_true",default=False,
                      help="Pass -v flag to view the learned centroids")
    parser.add_option("-p","--pixels",dest="pixels",action="store_true",default=False,
                      help="Pass -p flag to train the classifier on the raw pixels")
    parser.add_option("-m","--more_data",dest="more_data",action="store_true",default=False,
                      help="Pass -m flag to train the unsupervised algorithm on more unlabeled data")

    (opts,args) = parser.parse_args(args)

    if opts.seed:
        print "Setting random seed"
        np.random.seed(33)

    if opts.more_data:
        numTrain = 1000
    else:
        numTrain = 500

    if opts.grad=="perceptron":
        grad = util.perceptron
        maxIter = 300
    elif opts.grad=="logistic":
        grad = submission.logisticGradient
        maxIter = 300
    elif opts.grad=="hinge":
        grad = submission.hingeLossGradient
        maxIter = 300
    else:
        print "That loss function doesn't exist"
        import sys
        sys.exit(1)

    print "Using the %s loss function"%opts.grad

    run(grad,opts.view,opts.pixels,maxIter,numTrain)


if __name__=='__main__':
    main()
    
