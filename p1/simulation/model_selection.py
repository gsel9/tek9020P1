from itertools import chain, combinations

import numpy as np 
import pandas as pd 

from .splitting import train_test_split


def error_rate(y_true, y_pred):

	return sum(y_true != y_pred) / len(y_true)


def feature_combinations(n_elements):

	elem_set = np.arange(n_elements)

	combos = chain(*map(lambda x: combinations(elem_set, x), range(1, len(elem_set) + 1))) 

	return list(combos)


def select_optimal_features(model, X, y, path_to_results):

	opt_results, opt_features = feature_selection(model, X, y, path_to_results)

	pd.Series(opt_results, name="error_rate").to_csv(f"{path_to_results}_opt_results.csv")

	np.save(f"{path_to_results}_opt_features.npy", opt_features)


def feature_selection(model, X, y, path_to_results):
	"""Selects the optimal feature set."""

	feature_combos = feature_combinations(X.shape[1])

	results, best_score, opt_features = {}, -1, None 
	for combo in feature_combos:

		X_train, X_test, y_train, y_test = train_test_split(X[:, combo], y)

		model.fit(X_train, y_train)

		score = error_rate(y_test, model.predict(X_test))
		if score > best_score:

			best_score = score
			opt_features = combo

		results[("_").join([str(c) for c in combo])] = score

	return results, combo


def model_validation(X, y, model, path_to_results, results=None):

	X_train, X_test, y_train, y_test = train_test_split(X, y)

	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	np.save(f"{path_to_results}_y_pred.npy", y_pred)
	np.save(f"{path_to_results}_y_true.npy", y_test)

	if results is not None:
		results[model.name] = error_rate(y_test, y_pred)
	