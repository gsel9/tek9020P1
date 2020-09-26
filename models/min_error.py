

class MinErrorClassifier():

	def __init__(self):
		
		# Placeholders for the learned parameters. 
		self.W, self.w, self.b = [], [], []

	def _fit_gi(self, X, y, i):
		"""
		Args:
			i: Class index [0, inf].
		"""

		# TODO: Check axis
		mu_hat = np.mean(X, axis=1)

		# TODO: Check axis
		sigma_hat = np.mean((X - mu_hat) @ np.transpose(X - mu_hat), axis=1)
		sigma_hat_inv = np.linalg.inv(sigma_hat)

		self.W[i] = -0.5 * sigma_hat_inv
		self.w[i] = sigma_hat_inv @ mu_hat

		Pi = np.mean(y == i)
		self.b[i] = -0.5 * mu_hat.T @ sigma_hat_inv @ mu_hat - 0.5 * np.log(det(sigma_hat)) + np.log(Pi)
	
	# TODO: Vectorize?
	def g(self, x, i):
		"""
		Args:
			x: Sample vector.
			i: Class index [0, inf].
		"""

		return np.transpose(x) @ self.W[i] @ x + self.w[i].T @ x + self.b[i]

	def fit(self, X, y):

		for i in np.unique(y):

			X_i = X[y == i]
			y_i = y[y == i]

			# Learn per-class weights.
			self._fit_gi(X_i, y_i, i=i)

	def predict(X):
		
		# NOTE: Shuold probably be called per sample.
		if self.g(X, i=0) - self.g(X, i=1) >= 0:
			return 0

		return 1