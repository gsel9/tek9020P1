import numpy as np 

from data.load_data import from_file
from data.splitting import train_test_split

from models.min_error import MinErrorClassifier


def main():

	X, y = from_file("data/ds-1.txt")
	X_train, X_test, y_train, y_test = train_test_split(X, y)

	model = MinErrorClassifier()
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	print(np.mean(y_pred == y_test))


if __name__ == "__main__":
	main()
	