import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import strlearn as sl
from sklearn.cluster import KMeans


class BLS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, budget=0.5):
        self.budget = budget
        self.base_estimator = base_estimator

    def partial_fit(self, X, y, classes=None):
        np.random.seed(1410)
        # First train
        if not hasattr(self, "clf"):
            # Pierwszy chunk na pelnym
            self.clf = clone(self.base_estimator)

        # Get random subset
        limit = int(self.budget * len(y))
        idx = np.array(list(range(len(y))))
        selected = np.random.choice(idx, size=limit, replace=False)

        # Partial fit
        self.clf.partial_fit(X[selected], y[selected], classes)

    def predict(self, X):
        return self.clf.predict(X)


class ALS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, treshold=0.2):
        self.treshold = treshold
        self.base_estimator = base_estimator

    def partial_fit(self, X, y, classes=None):
        # First train
        if not hasattr(self, "clf"):
            # Pierwszy chunk na pelnym
            self.clf = clone(self.base_estimator).partial_fit(X, y, classes=classes)
            self.usage = []

        else:
            supports = np.abs(self.clf.predict_proba(X)[:, 0] - 0.5)
            selected = supports < self.treshold

            if np.sum(selected) > 0:
                self.clf.partial_fit(X[selected], y[selected], classes)

                score = sl.metrics.balanced_accuracy_score(
                    y[selected], self.clf.predict(X[selected])
                )

                # self.treshold = 0.5 - score / 2

            self.usage.append(np.sum(selected) / selected.shape)

    def predict(self, X):
        return self.clf.predict(X)


class BALS(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, treshold=0.2, budget=0.2):
        self.treshold = treshold
        self.budget = budget
        self.base_estimator = base_estimator

    def partial_fit(self, X, y, classes=None):
        # First train
        if not hasattr(self, "clf"):
            # Pierwszy chunk na pelnym
            self.clf = clone(self.base_estimator).partial_fit(X, y, classes=classes)
            self.usage = []

        else:
            supports = np.abs(self.clf.predict_proba(X)[:, 0] - 0.5)
            selected = supports < self.treshold

            if np.sum(selected) > 0:
                self.clf.partial_fit(X[selected], y[selected], classes)

                score = sl.metrics.balanced_accuracy_score(
                    y[selected], self.clf.predict(X[selected])
                )

                # self.treshold = 0.5 - score / 2

            self.usage.append(np.sum(selected) / selected.shape)

            # Get random subset
            limit = int(self.budget * len(y))
            idx = np.array(list(range(len(y))))
            selected = np.random.choice(idx, size=limit, replace=False)

            # Partial fit
            self.clf.partial_fit(X[selected], y[selected], classes)

    def predict(self, X):
        return self.clf.predict(X)
