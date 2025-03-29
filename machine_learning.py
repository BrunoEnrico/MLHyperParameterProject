from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from typing import Any
from sklearn.base import BaseEstimator
import numpy as np

class MachineLearning:
    def __init__(self):
        self.SEED = 301
        np.random.seed(self.SEED)

    @staticmethod
    def get_dummy_classifier(**kwargs) -> DummyClassifier:
        """
        Returns instance of dummy classifier.
        :param kwargs: Arguments for the dummy_classifier.
        :return: Dummy classifier instance.
        """
        return DummyClassifier(**kwargs)

    @staticmethod
    def get_cross_validate(model: BaseEstimator, feature: np.ndarray, target: np.ndarray, cv: Any, **kwargs):
        """
        Returns cross-validated estimator.
        :param model: Model instance.
        :param feature: Feature data
        :param target: Target data
        :param cv: Number of folds.

        :return: Cross-validated estimator.
        """
        return cross_validate(model, feature, target, cv=cv, **kwargs)

    @staticmethod
    def get_cross_validate_mean(model: dict, column: str) -> float:
        """
        Returns mean cross-validated estimator.
        :param model: Model instance
        :param column: Column name
        :return: Mean cross-validated estimator.
        """
        return model[column].mean()

    @staticmethod
    def get_std_cross_validate(model: dict, column: str) -> float:
        """
        Returns standard deviation cross-validated estimator.
        :param model: Model instance
        :param column: Column to calculate
        :return: Standard deviation of cross-validated estimator.
        """
        return model[column].std()


    @staticmethod
    def get_decision_tree_classifier(**kwargs) -> DecisionTreeClassifier:
        """
        Returns instance of decision tree classifier.
        :param kwargs: Arguments for the decision_tree_classifier.
        :return: Instance of decision tree classifier.
        """
        return DecisionTreeClassifier(**kwargs)


    def print_results(self, results, column: str) -> None:
        """
        Prints info of the results of a model
        :param results: Results of the model.
        :param column: column name.
        """
        mean = self.get_cross_validate_mean(results, column=column)
        std = self.get_std_cross_validate(results, column=column)
        print("Mean cross-validated estimator:", mean)
        print("Standard deviation cross-validated estimator:", ((mean - 2 * std) * 100), (mean + 2 * std) * 100)

    @staticmethod
    def get_groups_k_fold(**kwargs) -> GroupKFold:
        """
        Returns instance of groups k-fold.
        :return: Groups k-fold instance.
        """
        return GroupKFold(**kwargs)

    @staticmethod
    def get_svc(**kwargs) -> SVC:
        """
        Gets instance of SVC.
        :return: Instance of SVC
        """
        return SVC(**kwargs)

    @staticmethod
    def get_standard_scaler(**kwargs) -> StandardScaler:
        """
        Gets instance of standard scaler.
        :return: instance of standard scaler
        """
        return StandardScaler(**kwargs)

    @staticmethod
    def get_pipeline(args: list) -> Pipeline:
        """
        Gets pipeline
        :param args: List of transformer and classifier.
        :return: Pipeline object with the given transformer and classifier.
        """
        return Pipeline(args)