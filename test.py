from scipy import signal
import numpy as np


def computeDistance(p,q): 
	convOnes = np.ones(q.shape)
	p2 = signal.fftconvolve(p**2,convOnes, mode='same')
	pq = -2*(signal.fftconvolve(p, q, mode='same'))
	q2 = np.sum(q**2)
	return(np.sqrt(p2-pq+q2))

p = np.array([[1,2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
q = np.array([[1, 2], [3, 4]])
print(p)
print(computeDistance(p, q))
