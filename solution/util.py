'''
File: util.py
-------------
This file holds all the helper methods that
students can use for assn3
'''

import numpy as np
#import matplotlib.pyplot as plt

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
    Pass in an array of patches (or centroids) in order to view them as
    images.
    """
    view_helper(patches.reshape(patch_dim,patch_dim,-1),patches.shape[-1])


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
        from PIL import Image
        im = Image.fromarray(self.__img_data)
        im = im.resize((128,128),Image.BILINEAR)
        im.show()

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

def view_helper(patches,num):
    from PIL import Image

    xnum = int(np.sqrt(num))
    if xnum**2 == num:
        ynum = xnum
    else:
        ynum = xnum+1

    imDim = 50
    
    # rescale to be [0-255]
    patches = patches-np.min(patches)
    patches = 255*patches/np.max(patches)

    newpatches = np.empty((imDim,imDim,num))

    for p in range(num):
        patch = patches[:,:,p]
        im = Image.fromarray(patch)
        im = im.resize((imDim,imDim),Image.BILINEAR)
        newpatches[:,:,p] = np.asarray(im.convert('L'))

    patches = newpatches
    image = np.zeros(((imDim+1)*ynum+1,(imDim+1)*xnum+1))

    for i in range(ynum):
        for j in range(xnum):
            imnum = i*xnum+j
            if imnum>=num:
                break
            image[i*(imDim+1)+1:i*(imDim+1)+imDim+1, \
                  j*(imDim+1)+1:j*(imDim+1)+imDim+1] \
                  = patches[:,:,imnum]
    
    image = Image.fromarray(image)
    image.show()
