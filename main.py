import numpy as np 
import pandas as pd 

from data.load_data import from_file

from models.min_error import MinErrorClassifier
from models.less_squares import LessSquaresClassifier
from models.nearest_neighbour import NearNeighClassifier

from simulation.model_selection import (
	select_optimal_features, model_validation
)


def main():

	path_to_data = "data"
	path_to_results = "results"

	X, y = from_file("data/ds-1.txt")

	# Step 1:
	select_optimal_features(NearNeighClassifier(), X, y, path_to_results)

	# Step 2:
	models = [
		MinErrorClassifier(),
		NearNeighClassifier(),
		LessSquaresClassifier()
	]

	opt_features = np.load(f"{path_to_results}/opt_features.npy")

	results = {}
	for model in models:
		model_validation(X[:, opt_features], y, model, path_to_results, results=results)

	pd.Series(results).to_csv(f"{path_to_results}/model_validation.csv")


if __name__ == "__main__":
	main()
	