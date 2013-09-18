'''
File: util.py
-------------
This file holds all the helper methods that
students can use for assn3
'''

import numpy as np
import matplotlib.pyplot as plt

# Define a few global variables
image_x = 32 # width of image
image_y = 32 # height of image
patch_dim = 8 # height/width of a patch

def loadTrainImages():
    """
    Function: Load Train Images
    ---------------------
    Loads all *training* images from the dataset and returns them in a
    python list.
    """

    numTrainImages = 2000
    file_tag = 'train'
    image_list = load_helper(file_tag,numTrainImages)
    return image_list

def loadTestImages():
    """
    Function: Load Test Images
    --------------------------
    Loads all *testing* images from the dataset and returns them in a
    python list
    """

    numTestImages = 1000
    file_tag = 'test'
    image_list = load_helper(file_tag,numTestImages)
    return image_list

def viewPatches(patches):
    """
    Function: View Patches
    ----------------------
    Pass in a list of patches (or centroids) in order to view them as
    images.
    """

    # convert to list
    if type(patches).__name__=='ndarray':
        patches = patches.transpose().tolist()

    numToView = len(patches)
    x = y = int(np.sqrt(numToView))
    if x*y < numToView:
        y+=1

    for i in range(y):
        for j in range(x):
            if i*x+j >= len(patches):
                break
            ax = plt.subplot2grid((y,x),(i,j))
            ax.imshow(np.array(patches[i*x+j]).reshape((patch_dim,patch_dim)),
                      cmap = plt.get_cmap('gray'))
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    plt.subplots_adjust(wspace=-.5 ,hspace=0.2)            
    plt.show()

    

class Image(object):

    def __init__(self,data,label,patches):
        """
        Constructor
        -----------
        Takes image related data, called on image creation.
        """
        self.label = label # image label
        self.patches = patches.transpose().tolist()
        
        self.__img_data = data

    def view(self):
        """
        Function: View
        --------------
        Call function to view RGB image
        """
        fig = plt.imshow(self.__img_data)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()

    def getLabel(self):
        """
        Function: Label
        ---------------
        Returns label of image
        """
        return self.label

    def getPatches(self):
        """
        Function: Patches
        -----------------
        Returns list of patch vectors. Each patch length patch_size
        """
        return self.patches

###############################################
# Students don't need to read any code beyond # 
# this point!!                                #
###############################################

def load_helper(name,m):
    channels = 3
    patch_dim = 8
    patches_per_image = (image_x/patch_dim)*(image_y/patch_dim)

    images = np.fromfile('data/images_'+name+'.bin',dtype=np.uint8)
    images = images.reshape((m,image_x,image_y,channels))

    patches = np.fromfile('data/patches_'+name+'.bin',dtype=np.float32)
    patches = patches.reshape((patch_dim**2,-1))

    labels = np.fromfile('data/labels_'+name+'.bin',dtype=np.uint8)

    image_list = []
    for i in range(images.shape[0]):
        image_list.append(Image(images[i,...],labels[i],
          patches[:,i*patches_per_image:(i+1)*patches_per_image]))
    
    return image_list
