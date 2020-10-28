import numpy as np 
import pandas as pd 

from data.load_data import from_file
from data.splitting import train_test_split

from models.min_error import MinErrorClassifier
from models.less_squares import LessSquaresClassifier
from models.nearest_neighbour import NearNeighClassifier

from simulation.model_selection import (
	feature_selection_cv, feature_selection_summary
)


def select_optimal_features(model, X, y, path_to_results):

	opt_cv_results, opt_features = feature_selection_cv(model, X, y, cv=5,
														path_to_results=path_to_results)

	pd.DataFrame(opt_cv_results).to_csv(f"{path_to_results}/opt_cv_results.csv")

	np.save(f"{path_to_results}/opt_features.npy", opt_features)


def main():

	X, y = from_file("data/ds-1.txt")

	# Step 1:
	select_optimal_features(NearNeighClassifier(), X, y, "results/feature_selection_cv")

	feature_selection_summary("results/feature_selection_cv")

	# Step 2:
	models = [
		MinErrorClassifier(),
		NearNeighClassifier(),
		LessSquaresClassifier()
	]

	path_to_results = "results"

	opt_features = np.load(f"{path_to_results}/opt_features.npy")
	for model in models:

		X_train, X_test, y_train, y_test = train_test_split(X[:, opt_features], y)

		model.fit(X_train, y_train)

		np.save(f"{path_to_results}/{model.name}_y_pred.npy", model.predict(X_test))
		np.save(f"{path_to_results}/{model.name}_y_true.npy", y_test)	


if __name__ == "__main__":
	main()
	