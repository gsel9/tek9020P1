import numpy as np 


def train_test_split(X, y):

	# Deterministic splitting for comparability.  and test data the even numbered samples.

	# Training data is odd numbered samples.
	X_train = X[::2]
	y_train = y[::2]

	# Test data is odd numbered samples.
	X_test = X[1::2]
	y_test = y[1::2]

	# Sanity checks.
	assert np.shape(np.vstack([X_train, X_test])) == np.shape(X)
	assert np.isclose(len(y), len(np.append(y_train, y_test)))

	return X_train, X_test, y_train, y_test


if __name__ == "__main__":

	from load_data import from_file

	X, y = from_file("ds-1.txt")
	X_train, X_test, y_train, y_test = train_test_split(X, y)

	print(X[:10])
	print()
	print(X_train[:5])
	print()
	print(X_test[:5])
	