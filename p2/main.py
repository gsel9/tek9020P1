import numpy as np 
import matplotlib.pyplot as plt 

from skimage import io

from min_error import MinErrorClassifier


def normalize_rgb(X):

	scale = np.sum(X, axis=1)
	X_norm = np.array([X[:, 0] / scale, X[:, 1] / scale])

	return np.transpose(X_norm)


def load_train_data():

	I1_fg1 = io.imread(f"../data/p2/train_fg1.png")[:, :, :3]
	I1_fg2 = io.imread(f"../data/p2/train_fg2.png")[:, :, :3]
	I1_bg1 = io.imread(f"../data/p2/train_bg1.png")[:, :, :3]

	classes = [0, 0, 1]

	X_train, y_train = [], []
	for j, I in enumerate([I1_fg1, I1_fg2, I1_bg1]):
		
		N, T, C = np.shape(I)

		y_train.extend(np.ones(N * T) * classes[j])
		X_train.append(np.reshape(I, (N * T, C)))

	return normalize_rgb(np.vstack(X_train)), np.array(y_train)


def load_test_data(path_to_file):
	
	I = io.imread(path_to_file)[:, :, :3]
	
	N, T, C = np.shape(I)

	return normalize_rgb(np.reshape(I, (N * T, C)))


def train_test_model():

	X_train, y_train = load_train_data()
	
	clf = MinErrorClassifier()
	clf.fit(X_train, y_train)

	X_train = load_test_data("../data/p2/train.png")
	np.save("results/y_val_pred.npy", clf.predict(X_train))

	X_test = load_test_data("../data/p2/test.png")
	np.save("results/y_test_pred.npy", clf.predict(X_test))
	

def plot_segmentation(I, y_pred, path_to_fig):
	
	M = y_pred.copy()
	M[y_pred == 0] = 1
	M[y_pred == 1] = 255
	M = np.reshape(M, (I.shape[0], I.shape[1], 1))
	
	plt.figure()
	plt.imshow(I * M)
	plt.xticks([])
	plt.yticks([])
	plt.savefig(path_to_fig)


if __name__ == "__main__":
	
	#train_test_model()

	I = io.imread("../data/p2/train.png")[:, :, :3]
	y_pred = np.load("results/y_val_pred.npy")
	plot_segmentation(I, y_pred, "results/I_val_seg.pdf")

	I = io.imread("../data/p2/test.png")[:, :, :3]
	y_pred = np.load("results/y_test_pred.npy")
	plot_segmentation(I, y_pred, "results/I_test_seg.pdf")
