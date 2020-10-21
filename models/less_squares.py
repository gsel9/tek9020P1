import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class LessSquaresClassifier(BaseEstimator, ClassifierMixin):
    """
    """

    def __init__(self):

        # Placeholders for the learned parameters of each class.
        self.a = None, None

    # TODO: Returna  dict of whatever params is used for the clf.
    def get_params(self, deep=True):

        return  # {"alpha": self.alpha, "recursive": self.recursive}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self

    @property
    def n_classes(self):

        return int(len(self.classes_))

    @property
    def n_features(self):

        return int(self.X_.shape[1])

    @property
    def n_objects(self):

        return int(self.X_.shape[0])

    def fit(self, X, y):
        """Train the model."""

        # Check that X and y have correct shape.
        X, y = check_X_y(X, y)

        # Store the classes seen during fit.
        self.classes_ = unique_labels(y)
        if self.n_classes > 2:
            raise ValueError("Target classes should be binary.")

        self.X_ = X
        self.y_ = y

        # Find the extended weight vector
        self.X_ext = np.c_[np.ones(self.n_objects), X]
        self.b = np.array([1 if y_ == 0 else -1 for y_ in y], dtype=int)
        yy = (np.transpose(self.X_ext) @ self.X_ext)
        self.a = np.linalg.inv(yy) @  np.transpose(self.X_ext) @ self.b

        # Return the classifier.
        return self

    def g(self, x):
        """Discriminant function."""

        x_ext = np.append(1, x)
        g = np.transpose(self.a) @ x_ext

        return float(g)

    def predict(self, X):

        # Check is fit had been called.
        # check_is_fitted(self) #Can't get this to work

        # Input validation.
        X = check_array(X)

        return np.array([0 if self.g(x) >= 0 else 1 for x in X], dtype=int)

