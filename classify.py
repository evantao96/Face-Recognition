import math
import numpy as np
from scipy import signal, ndimage
from scipy.io import loadmat
from sklearn import svm
import pickle
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize
import os

# returns value of a gabor at position (i, j) and orientation theta
def computeGabor(i, j, theta): 
    stdev = 10.24 # standard deviation
    wave = 3.6 # wavelength
    x = i*math.cos(theta) + j*math.sin(theta)
    y = -i*math.sin(theta) + j*math.cos(theta)
    E = math.exp(-1*((x**2) + (y**2))/(2*stdev))*math.cos(2*math.pi*x/wave)
    return(E)

# computes Euclidean distance between an image and a patch
def computeDistance(p,q): 
    convOnes = np.ones(q.shape)
    p2 = signal.fftconvolve(p**2,convOnes, mode='same')
    pq = -2*(signal.fftconvolve(p, q, mode='same'))
    q2 = np.sum(q**2)
    return(np.sqrt(p2-pq+q2))

# returns responses of an image and 100 patches
def computeResponses(image, patches): 

    # create gabor filters
    filterSize = 9 # filter dimensions
    filterSizeL = -(filterSize//2)
    filterSizeR = filterSize//2

    gabor = [normalize([[computeGabor(i, j, k*math.pi/4) for i in range(filterSizeL, filterSizeR+1)] for j in range(filterSizeL, filterSizeR+1)]) for k in range(4)]

    # convolve image and filters 
    convImage = [normalize(signal.fftconvolve(image, gabor[i], mode='same')) for i in range(4)]

    # compute local maximums
    maxImage = [normalize(ndimage.maximum_filter(convImage[i], size=8)) for i in range(4)]

    # compute euclidean distances from patches
    patchOutputs = [[computeDistance(maxImage[i], patches[j][i, :, :]) for i in range(4)] for j in range(len(patches))]

    # compute global maximums
    maxOutputs = [max([ndimage.maximum(patchOutputs[j][i]) for i in range(4)]) for j in range(len(patches))]

    return(maxOutputs)

# load patches
print('Loading patches...   ')
m = loadmat('./universal_patch_set.mat')
patches = [patch.reshape(4, 4, 4) for patch in m['patches'][0][1].T[0:100]]

# compute 20 positive training responses
print('Training on 20 faces...')
directory = './training/train_positive/'
X1 = [computeResponses(mpimg.imread(directory+filename)[:,:,1], patches) for filename in os.listdir(directory) if filename.endswith(".jpg")]

# compute 20 negative training responses
print('Training on 20 non-faces...')
directory = './training/train_negative/'
X2 = [computeResponses(mpimg.imread(directory+filename)[:,:,1], patches) for filename in os.listdir(directory) if filename.endswith(".jpg")]

# classify
X = np.concatenate((X1, X2))
y = np.array([1]*20 + [0]*20)
clf = svm.SVC()
clf.fit(X, y)

model_pkl_file = "clf_classifier_model.pkl"

with open(model_pkl_file, 'wb') as file:
    pickle.dump(clf, file)

# compute 20 positive test responses
print('Testing on 20 faces...')
directory = './test/test_positive/'
results1 = [clf.predict(np.array(computeResponses(mpimg.imread(directory+filename)[:,:,1],patches)).reshape(1,-1))[0] for filename in os.listdir(directory) if filename.endswith(".jpg")]
# print(results1)
print('Accuracy: %d/20' % (sum(results1)))

# compute 20 negative test responses
print('Testing on 20 non-faces...')
directory = './test/test_negative/'
results2 = [clf.predict(np.array(computeResponses(mpimg.imread(directory+filename)[:,:,1],patches)).reshape(1,-1))[0] for filename in os.listdir(directory) if filename.endswith(".jpg")]
# print(results2)
print('Accuracy: %d/20' % (20-sum(results2)))