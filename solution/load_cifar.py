import numpy as np
import matplotlib.pyplot as plt

label_dict = {'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,
              'frog':6,'horse':7,'ship':8,'truck':9}

def unpickle(file):
    """
    Unpickles file and loads into python dict
    """
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_file(file_name):
    """
    Loads a given CIFAR batch file. Returns labels 
    as one dimensional numpy array and data as 4 
    dimensional numpy array where dimensions are 
    [numsamples, image height, image width, RGB channels]
    """
    data_dict = unpickle(file_name)

    # reshape data
    data = data_dict['data']
    data = data.reshape(data.shape[0],3,32,32)
    data = data.transpose(0,2,3,1)

    labels = np.array(data_dict['labels'],dtype=np.uint8)

    return data,labels


