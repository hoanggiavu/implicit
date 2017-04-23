"""
Extension of benfred's implicit library.
"""
from als import AlternatingLeastSquares
from . import _als

import itertools
import logging
import time

import numpy as np


class ImplicitALS(AlternatingLeastSquares):

    def __init__(self, factors=100, regularization=0.01,
                 alpha=1.0, dtype=np.float64,
                 use_native=True, use_cg=True,
                 iterations=15, calculate_training_loss=False, num_threads=0):
        super(ImplicitALS, self).__init__(factors, regularization, dtype, use_native, use_cg,
                                          iterations, calculate_training_loss, num_threads)
        self.alpha = alpha

    def fit(self, item_users):
        Ciu = item_users.copy()
        Ciu.data *= self.alpha
        super(ImplicitALS, self).fit(Ciu)

    def predict(self, user_idx=None, item_idx=None):
        """
        Calculate the predicted "rating" matrix for given users and items.

        :param user_idx: Index of users. If None, all user indices in the training set will be used.
        :param item_idx: Index of items. If None, all item indices in the training set will be used.
        :return: the predicted rating matrix R wherein R[u,i] is the predicted rating for user u and item i.
        """
        if user_idx is None:
            u_factors = self.user_factors
        else:
            u_factors = self.user_factors[user_idx]

        if item_idx is None:
            i_factors = self.item_factors
        else:
            i_factors = self.item_factors[item_idx]
        # TODO for large number of users/items, computing this matrix product is infeasible
        return u_factors.dot(i_factors)

    def mse(self, user_items):
        """
        A memory-efficient way to compute the Mean Square Error
        :param user_items:
        :return:
        """
        Cui = user_items.copy()
        Cui.data *= self.alpha
        return _als.calculate_loss(Cui, self.user_factors, self.item_factors,
                                   self.regularization, num_threads=self.num_threads)
