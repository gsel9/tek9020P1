

def tpr(y_true, y_pred):
	"""Performance metric is accuracy."""

	return sum(y_true != y_pred) / len(y_true)