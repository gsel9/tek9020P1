import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class NearNeighClassifier(BaseEstimator, ClassifierMixin):
	"""
	"""

	def __init__(self):
		pass

	# TODO: Returna  dict of whatever params is used for the clf.
	def get_params(self, deep=True):
		
		return #{"alpha": self.alpha, "recursive": self.recursive}

	def set_params(self, **parameters):

		for parameter, value in parameters.items():
			setattr(self, parameter, value)

		return self

	@property 
	def n_classes(self):

		return int(len(self.classes_))

	@property 
	def n_features(self):

		return int(self.X_.shape[1])


	def fit(self, X, y):
		"""Check that inputs are reasonable and input training data"""

		# Check that X and y have correct shape.
		X, y = check_X_y(X, y)

		# Store the classes seen during fit.
		self.classes_ = unique_labels(y)
		if self.n_classes > 2:
			raise ValueError("Target classes should be binary.")

		self.X = X
		self.y = y

		# Return the classifier.
		return self


	def _predict(self, x_test):

		"""Predict each object"""

		dist = [np.linalg.norm(x_test - x_train) for x_train in self.X]
		idx_min = dist.index(min(dist))

		return self.y[idx_min]


	def predict(self, X):

		# Check is fit had been called.
		#check_is_fitted(self) #Can't get this to work

		# Input validation.
		X = check_array(X)

		return np.array([self._predict(x_test) for x_test in X], dtype=int)

