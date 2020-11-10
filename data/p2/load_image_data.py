import numpy as np 
from skimage import io


def load_png(path_to_image):

	# NOTE: Ignore opacity and limit to only RGB channels.
	I = io.imread(path_to_image)[:, :, :3]

	# Each pixel is a sample.
	return np.transpose([I[:, :, c].ravel() for c in range(3)])


if __name__ == "__main__":

	X = load_png("Bilde1.png")

	print(X.shape)

	#import matplotlib.pyplot as plt 
	#plt.figure()
	#plt.imshow(I[:, :, :3])
	#plt.show()
