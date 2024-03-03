import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import signal, ndimage
from scipy.io import loadmat
from sklearn.preprocessing import normalize

# returns value of a gabor at position (i, j), orientation theta, filter dimensions size x size and scaling factor div
def computeGabor(i, j, theta, size, div): 
    wave = size*2/div # wavelength
    variance = (wave*0.8)**2 # standard deviation 
    gamma = 0.3 # spatial aspect ratio: 0.23 < gamma < 0.92

    if math.sqrt(i**2+j**2) > size/2: # position is out of the receptive field
        E = 0
    else: # position is in the receptive field
        x = i*math.cos(theta)-j*math.sin(theta)
        y = i*math.sin(theta)+j*math.cos(theta)
        E = math.exp(-1*((x**2)+(gamma**2)+(y**2))/(2*variance))*math.cos(2*math.pi*x/wave)
    return(E)

# read face
image = mpimg.imread('./training/train_positive/image_0006.jpg')[:,:,1]

# create gabor filters
gabor = []
for filterSize in range(7, 41, 2): # filter dimensions
    filterDiv = (filterSize//-40)+4.175
    filterSizeL = -(filterSize//2)
    filterSizeR = filterSize//2
    gabor += [normalize([[computeGabor(i, j, (k-1)*math.pi/4, filterSize, filterDiv) for i in range(filterSizeL, filterSizeR+1)] for j in range(filterSizeL, filterSizeR+1)]) for k in range(4)]

# # convolve image and filters 
# convImage = [normalize(signal.fftconvolve(image, gabor[i], mode='same')) for i in range(4)]

# # compute local maximums
# maxImage = [normalize(ndimage.maximum_filter(convImage[i], size=8)) for i in range(4)]

# # load patches
# m = loadmat('./universal_patch_set.mat')
# patches = [patch.reshape(4, 4, 4) for patch in m['patches'][0][1].T[0:100]]

# print results 
np.set_printoptions(precision=2, threshold=100, edgeitems=2)
print("\nImage: ")
print(image)
print("\nImage dimensions: ")
print(image.shape)
print("\nGabor filter, theta = 0: ")
print(gabor[0])
print("\nGabor filter dimensions: ")
print(gabor[0].shape)
# print("\nPatch: ")
# print(patches[0][0,:,:])
# print("\nPatch dimensions: ")
# print(patches[0][0,:,:].shape)

# plot results
fig, ax = plt.subplots(17, 2)
ax[0,0].imshow(image, cmap='gray')
ax[0,0].set_title('Image')
for i in range(4): 
	ax[i,0].axis('off')

ax[0,1].set_title('Gabor filters')
for i in range(0, 68, 4):
	ax[(i//4),1].imshow(gabor[i], cmap='gray')
	ax[(i//4),1].axis('off')

# ax[0,2].set_title('Convolved images')
# for i in range(4): 
# 	ax[i,2].imshow(convImage[i], cmap='gray')
# 	ax[i,2].axis('off')

# ax[0,3].set_title('Local maximums')
# for i in range(4): 
# 	ax[i,3].imshow(maxImage[i], cmap='gray')
# 	ax[i,3].axis('off')

# ax[0,4].set_title('Patches')
# for i in range(4): 
# 	ax[i,4].imshow(patches[i][0,:,:], cmap='gray')
# 	ax[i,4].axis('off')

# ax[0,5].set_title('Patches')
# for i in range(5,9): 
# 	ax[i-5,5].imshow(patches[i][0,:,:], cmap='gray')
# 	ax[i-5,5].axis('off')

plt.show()
