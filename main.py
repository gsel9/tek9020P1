import numpy as np 
import pandas as pd 

from data.load_data import from_file

from models.min_error import MinErrorClassifier
from models.less_squares import LessSquaresClassifier
from models.nearest_neighbour import NearNeighClassifier

from simulation.model_selection import (
	select_optimal_features, model_validation
)


from itertools import product
import matplotlib.pyplot as plt


def model_performance():

	path_to_data = "data"
	path_to_results = "results"

	for dataset in ["ds-2.txt"]:

		X, y = from_file(f"{path_to_data}/{dataset}")

		# Step 1:
		#select_optimal_features(NearNeighClassifier(), X, y, f"{path_to_results}/{dataset}")

		# Step 2:
		models = [
			#MinErrorClassifier(),
			NearNeighClassifier(),
			#LessSquaresClassifier()
		]

		opt_features = np.load(f"{path_to_results}/{dataset}_opt_features.npy")
		
		results = {}
		for model in models:
			model_validation(X[:, opt_features], y, model, 
							 f"{path_to_results}/{dataset}_{model.name}", 
							 results=results)

		pd.Series(results, name="error_rate").to_csv(f"{path_to_results}/{dataset}_model_validation.csv")


def plot_decision_surface():

	from utils.plot_utils import plot_config, set_fig_size, set_arrowed_spines

	X, y = from_file("data/ds-2.txt")

	opt_features = np.load("results/ds-2.txt_opt_features.npy")
	opt_features = [0, 1]

	clf = NearNeighClassifier()
	clf.fit(X[:, opt_features], y)

	# Plotting decision regions
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	f, axarr = plt.subplots(1, 1, figsize=set_fig_size(430, fraction=0.7))

	axarr.contourf(xx, yy, Z, alpha=0.4)
	axarr.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

	axarr.set_xlabel(r"$x_1$")
	axarr.set_ylabel(r"$x_2$")
	#axarr.set_title(r'KNN ($k=1$)')

	axarr.spines['top'].set_visible(False)
	axarr.spines['right'].set_visible(False)

	#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

	set_arrowed_spines(f, axarr)

	plt.tight_layout()

	f.savefig('results/knn_decision_surface.pdf', format='pdf', 
			  bbox_inches='tight', transparent=True)

	#plt.show()


if __name__ == "__main__":
	#model_performance()
	plot_decision_surface()
	 