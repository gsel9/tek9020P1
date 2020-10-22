from itertools import chain, combinations

import numpy as np 

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

from .metrics import tpr


def feature_combinations(n_elements):

	elem_set = np.arange(n_elements)

	combos = chain(*map(lambda x: combinations(elem_set, x), range(1, len(elem_set) + 1))) 

	return list(combos)


def feature_selection_cv(model, X, y, cv=5):
	"""Selects the optimal feature set using k-fold cross-validation.
	"""

	feature_combos = feature_combinations(X.shape[1])

	opt_cv_results, opt_features = None, None

	opt_score = -1
	for combo in feature_combos:

		cv_results = cross_validate(model, X[:, combo], y, scoring=make_scorer(tpr))

		if np.mean(cv_results["test_score"]) > opt_score:

			opt_score = np.mean(cv_results["test_score"])
			opt_cv_results = cv_results
			opt_features = combo

	return opt_cv_results, opt_features
