import os 

from itertools import chain, combinations
from collections import defaultdict

import numpy as np 
import pandas as pd

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate

from .metrics import tpr


def feature_selection_summary(path_to_results):

	results = defaultdict(list)
	for fname in os.listdir(path_to_results):

		if fname.endswith("csv") and "results_summary" not in fname:

			exp_results = pd.read_csv(f"{path_to_results}/{fname}", index_col=0)

			results["run_id"].append(("_").join(fname.split("_")[1:]).split(".")[0])
			results["std_score"].append(np.std(exp_results.test_score))
			results["mean_score"].append(np.mean(exp_results.test_score))
			
	pd.DataFrame(results).to_csv(f"{path_to_results}/results_summary.csv")


def feature_combinations(n_elements):

	elem_set = np.arange(n_elements)

	combos = chain(*map(lambda x: combinations(elem_set, x), range(1, len(elem_set) + 1))) 

	return list(combos)


def feature_selection_cv(model, X, y, path_to_results=None, cv=5):
	"""Selects the optimal feature set using k-fold cross-validation.
	"""

	feature_combos = feature_combinations(X.shape[1])

	opt_cv_results, opt_features = None, None

	opt_score = -1
	for combo in feature_combos:

		cv_results = cross_validate(model, X[:, combo], y, scoring=make_scorer(tpr))

		if path_to_results is not None:

			fname = ("_").join([str(c) for c in combo])

			pd.DataFrame(cv_results).to_csv(f"{path_to_results}/{model.name}_{fname}.csv")

		if np.mean(cv_results["test_score"]) > opt_score:

			opt_score = np.mean(cv_results["test_score"])
			opt_cv_results = cv_results
			opt_features = combo

	return opt_cv_results, opt_features
	

def model_performane_estimate(cv=None):

	model.train(X_train, y_train)

	if cv is None:
		return cross_validate(model, X_test, y_test, scoring=make_scorer(tpr))

	y_pred = model.predict(X_test)

	return tpr(y_true, y_pred)