

class MSClassifier():

	def __init__(self):	

		self.w, self.b = None, None 

	def fit(self, X, y):

		# Normal equations.
		N = np.linalg.inv(X.T @ X) @ X

		self.w = N @ y

	def predict(self, x):
		
		return self.w.T @ x 
