

import os

STARTCODE = "### YOUR CODE HERE ###"
ENDCODE = "### END CODE ###"

def strip_file(filename,inpath,outpath):

    print inpath+filename
    solutionFile = open(inpath+'/'+filename,'r')
    starterFile = open(outpath+'/'+filename,'w')
    lines = solutionFile.readlines()

    skip = False
    for l in lines:
        if not skip:
            starterFile.write(l)
        if STARTCODE in l:
            skip = True
        if ENDCODE in l:
            skip = False

    solutionFile.close()
    starterFile.close()

def create_starter_zip(path):

    #files = os.listdir(path)

    files = ['util.py', 'evaluator.py', 'featureLearner.py', 'classifier.py']
    starterPath = 'image-classifier-starter'
    os.mkdir(starterPath)

    for f in files:
        strip_file(f,path,starterPath)
    
    os.system('cp -r '+path+'/data '+starterPath)
    # os.system('zip -r '+starterPath+'.zip '+starterPath)

    # remove starter code, leave only archive
    # os.system('rm -rf '+starterPath)

    print "Zipped exercise %s"%path

if __name__=='__main__':

    # list for now
    exDirs = ['solution']

    for d in exDirs:
        create_starter_zip(d)










