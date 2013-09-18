import numpy as np
import matplotlib.pyplot as plt

def extract_patches(data,patch_dim):
    """
    Extracts square patches of patch_dim by patch_dim from
    the data.  Returns a patch_dim^2 by numpatches matrix.
    """
    
    m = data.shape[0]
    im_x = data.shape[1]
    im_y = data.shape[2]
    
    assert im_x%float(patch_dim)==0 and im_y%float(patch_dim)==0, \
    "patch_size must divide x and y dimensions of image"

    numpatchs = m*(im_x/patch_dim)*(im_y/patch_dim)
    patch_size = patch_dim**2

    patches = np.empty((patch_size,numpatchs))
    p=0
    for i in range(data.shape[0]):
        image = data[i,...]
        for x in np.r_[0:im_x:patch_dim]:
            for y in np.r_[0:im_y:patch_dim]:
                patch = image[x:x+patch_dim,y:y+patch_dim]
                patches[:,p] = patch.ravel()
                p+=1
                
    return patches

def normalize(data):
    """
    Subtract DC component (mean) from each patch,
    and contrast normalize (divide by std dev).
    """

    p_means = np.mean(data,axis=0)
    p_vars = np.var(data,axis=0)

    # subtract dc component
    data = data-p_means

    # contrast normalize 
    data = data/np.sqrt(p_vars+10) # plus 10 to account for small variances
    
    return data

def whiten(data):
    """
    Makes ZCA whitening matrix (U*(V+epsilon*I)^(-1/2)*U^T),
    where U is the matrix of orthogonal eigenvectors and V 
    the corresponding eigenvalues from eigenvalue decomposition
    of the covariance matrix for the data.
    """

    from numpy.linalg import eig

    eps = 0.01
    
    # covariance matrix
    Sigma = np.cov(data)

    # eigenvalue decomposition
    V,U = eig(Sigma)
    
    W = U.dot(np.diag((V+eps)**(-0.5)).dot(U.transpose()))

    return W

def process(data,patch_dim,W=None):
    """
    Extracts patches from data and whitens the patches.
    Returns whitened patches and whitening matrix.
    """

    # pull patches from image
    patches = extract_patches(data,patch_dim)

    # remove dc component and contrast normalize
    patches = normalize(patches)

    if W is None:
        # build whitening matrix
        W = whiten(patches)

    # apply whitening
    patches = W.dot(patches)

    return patches,W

def rgb2gray(rgb):
    """
    Convert rgb image to grayscale
    """
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def save_data(data,patches,labels,name,patch_dim,size=None):
    """
    Saves images and corresponding whitened patches
    in a format that makes sense for the images class.
    """

    patchPerIm = (data.shape[1]*data.shape[2])/(patch_dim**2)

    if size is None:
        size = labels.shape[0]

    img_f = open('data/images_'+name,'w')
    patches_f = open('data/patches_'+name,'w')
    label_f = open('data/labels_'+name,'w')

    patches = patches.astype(np.float32)

    data[:size,...].tofile(img_f)
    patches[:,:(size*patchPerIm)].tofile(patches_f)
    labels[:size].tofile(label_f)

    img_f.close()
    patches_f.close()
    label_f.close()

def save_whiten_mat(W):
    """
    Save whitening matrix. (Probably don't need this funtion)
    """
    w_id = open('data/W.bin','w')
    W.tofile(w_id)
    w_id.close()

def load_cifar_images(filename):   
    """
    Specific to CIFAR dataset
    Loads data from file, returns color, grayscale images
    and labels.
    """

    from load_cifar import load_file
    from load_cifar import label_dict

    data,labels = load_file(filename)

    # two classes to keep
    class0 = label_dict['airplane']
    class1 = label_dict['bird']
    # remove all but two classes
    keep = np.logical_or(labels==class0,labels==class1)
    data = data[keep,...]
    labels = labels[keep]
    # set labels to 0 or 1
    labels[labels==class0]=0
    labels[labels==class1]=1

    # rgb -> grayscale
    gray_data = rgb2gray(data)
    return data,gray_data,labels

if __name__=="__main__":

    trainSize = 2000
    testSize = 1000

    train_file = 'cifar-10-batches-py/data_batch_1'
    data_train,gray_data_train,labels_train = load_cifar_images(train_file)
    
    test_file = 'cifar-10-batches-py/test_batch'
    data_test,gray_data_test,labels_test = load_cifar_images(test_file)

    print "Training set size is %d"%data_train.shape[0]
    print "Test set size is %d"%data_test.shape[0]
    # view an image in gray scale
    # image = 1 # image num
    # plt.imshow(gray_data[image,...], cmap = plt.get_cmap('gray'))
    # plt.show()

    patch_dim = 8
    
    # process training data
    patches_train,W = process(gray_data_train,patch_dim)

    # process test data
    patches_test,_ = process(gray_data_test,patch_dim,W=W)

    save_data(data_train,patches_train,labels_train,'train.bin',patch_dim,size=trainSize)
    save_data(data_test,patches_test,labels_test,'test.bin',patch_dim,size=testSize)



