import math
import numpy as np
from scipy import signal, ndimage
from scipy.io import loadmat
from sklearn import svm
import pickle
import matplotlib.image as mpimg
from sklearn.preprocessing import normalize
import os

# returns value of a gabor at position (i, j), orientation theta, filter dimensions size x size and scaling factor div
def computeGabor(i, j, theta, size, div): 
    wave = size*2/div # wavelength
    variance = (wave*0.8)**2 # standard deviation 
    gamma = 0.3 # spatial aspect ratio: 0.23 < gamma < 0.92

    if sqrt(i**2+j**2) > size/2: # position is out of the receptive field
        E = 0
    else: # position is in the receptive field
        x = i*math.cos(theta)-j*math.sin(theta)
        y = i*math.sin(theta)+j*math.cos(theta)
        E = math.exp(-1*((x**2)+(gamma**2)+(y**2))/(2*variance))*math.cos(2*math.pi*x/wave)
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
    filterDiv = 3.8 
    filterSizeL = -(filterSize//2)
    filterSizeR = filterSize//2

    gabor = [normalize([[computeGabor(i, j, (k-1)*math.pi/4, filterSize, filterDiv) for i in range(filterSizeL, filterSizeR+1)] for j in range(filterSizeL, filterSizeR+1)]) for k in range(4)]

    # convolve image and filters 
    convImage = [normalize(signal.fftconvolve(image, gabor[i], mode='same')) for i in range(4)]

    # compute local maximums
    maxImage = [normalize(ndimage.maximum_filter(convImage[i], size=8)) for i in range(4)]

    # compute euclidean distances from patches
    patchOutputs = [[computeDistance(maxImage[i], patches[j][i, :, :]) for i in range(4)] for j in range(len(patches))]

    # compute global maximums
    maxOutputs = [max([ndimage.maximum(patchOutputs[j][i]) for i in range(4)]) for j in range(len(patches))]

    return(maxOutputs)