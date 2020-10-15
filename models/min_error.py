import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class MinErrorClassifier(BaseEstimator, ClassifierMixin):
	"""
	"""

	def __init__(self):
		
		# Placeholders for the learned parameters of each class. 
		self.W, self.w, self.b = None, None, None 

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

	def _fit(self, i):

		X_i = self.X_[self.y_ == i]
		y_i = self.y_[self.y_ == i]

		P_i = np.mean(self.y_ == i)

		# ML estimate for feature-wise mean.
		mu_hat = np.mean(X_i, axis=0)

		# ML estimate for feature-wise co-variance matrix.
		sigma_hat = np.transpose(X_i - mu_hat) @ (X_i - mu_hat) / X_i.shape[0]
		sigma_hat_inv = np.linalg.inv(sigma_hat)

		self.W[i, :, :] = -0.5 * sigma_hat_inv
		self.w[i, :] = sigma_hat_inv @ mu_hat

		quad = mu_hat.T @ sigma_hat_inv @ mu_hat

		# ERROR: Check if should be abs(det(C)) or only det()?
		log_det = np.log(abs(np.linalg.det(sigma_hat)))

		self.b[i] = -0.5 * quad - 0.5 * log_det + np.log(P_i)

	def fit(self, X, y):
		"""Train the model."""

		# Check that X and y have correct shape.
		X, y = check_X_y(X, y)

		# Store the classes seen during fit.
		self.classes_ = unique_labels(y)
		if self.n_classes > 2:
			raise ValueError("Target classes should be binary.")

		self.X_ = X
		self.y_ = y

		self.W = np.zeros((self.n_classes, self.n_features, self.n_features), dtype=float)
		self.w = np.zeros((self.n_classes, self.n_features), dtype=float)
		self.b = np.zeros(self.n_classes, dtype=float)

		# Learn parameters per-class.
		for i in self.classes_:
			self._fit(i=i)

		# Retrun the classifier.
		return self

	def g(self, x):
		"""Discriminant function."""

		g1 = np.transpose(x) @ self.W[0] @ x + np.transpose(self.w[0]) @ x + self.b[0]
		g2 = np.transpose(x) @ self.W[1] @ x + np.transpose(self.w[1]) @ x + self.b[1]

		return float(g1 - g2)

	def predict(self, X):

		# Check is fit had been called.
		check_is_fitted(self)

		# Input validation.
		X = check_array(X)

		return np.array([0 if self.g(x) >= 0 else 1 for x in X], dtype=int)

