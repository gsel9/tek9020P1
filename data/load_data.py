import numpy as np 
import pandas as pd 


def from_file(path_to_file):

	df = pd.read_csv(path_to_file, delim_whitespace=True, header=None)

	# First column is target label and remaining columns are features.
	y = np.array(df.iloc[:, 0].values, dtype=int)
	X = np.array(df.iloc[:, 1:], dtype=float)

	# Shift class labels from 1/2 -> 0/1.
	y = y - 1

	# Sanity check.
	assert np.array_equal(np.unique(y), [0, 1])

	return X, y


if __name__ == "__main__":

	X, y = from_file("ds-1.txt")
	print(X.shape, y.shape)

	X, y = from_file("ds-2.txt")
	print(X.shape, y.shape)

	X, y = from_file("ds-3.txt")
	print(X.shape, y.shape)
