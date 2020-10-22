import numpy as np 
import pandas as pd 

from data.load_data import from_file
from data.splitting import train_test_split

from models.min_error import MinErrorClassifier

from simulation.model_selection import feature_selection_cv


def select_optimal_features(model, X, y, path_to_results):

	cv_results, opt_features = feature_selection_cv(model, X, y, cv=5)

	pd.DataFrame(cv_results).to_csv(f"{path_to_results}/cv_results.csv")

	np.save(f"{path_to_results}/opt_features.npy", opt_features)


def main():

	X, y = from_file("data/ds-1.txt")

	model = MinErrorClassifier()

	select_optimal_features(model, X, y, "results")


if __name__ == "__main__":
	main()
	