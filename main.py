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

	for dataset in ["ds-1.txt", "ds-2.txt", "ds-3.txt"]:

		X, y = from_file(f"{path_to_data}/{dataset}")

		# Step 1:
		select_optimal_features(NearNeighClassifier(), X, y, f"{path_to_results}/{dataset}")

		# Step 2:
		models = [
			MinErrorClassifier(),
			NearNeighClassifier(),
			LessSquaresClassifier()
		]

		opt_features = np.load(f"{path_to_results}/{dataset}_opt_features.npy")

		results = {}
		for model in models:
			model_validation(X[:, opt_features], y, model, 
							 f"{path_to_results}/{dataset}_{model.name}", 
							 results=results)

		pd.Series(results, name="error_rate").to_csv(f"{path_to_results}/{dataset}_model_validation.csv")


if __name__ == "__main__":
	main()
	