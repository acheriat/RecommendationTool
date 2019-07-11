"""
A Simple  Implementation of Matrix Factorization.
Inspired by : https://github.com/albertauyeung/matrix-factorization-in-python/blob/master/mf.ipynb
"""

import numpy as np
import params


class MatrixFactorization:
    """
    Matrix Factorization algorithm in Python, using stochastic gradient descent.

    An article with detailed explanation of the algorithm can be found at
    http://www.albertauyeung.com/post/python-matrix-factorization/.
    """

    def __init__(self, r):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - r (ndarray)   : user-item rating matrix
        - k (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        - Seed (Int)    : Random seed used to initialize the pseudo-random number generator.
                          It used to reproduce the same results in unit tests (random effect)
        """

        self.r_matrix = r
        self.num_users, self.num_items = self.r_matrix.shape
        self.num_latfactors = params.k
        self.alpha = params.alpha
        self.beta = params.beta
        self.iterations = params.iterations
        self.seed = params.seed

        # Initialize user and item latent factors matrix
        rng = np.random.RandomState(self.seed)
        self.user_latfact_matrix = rng.normal(
            scale=1.0 / self.num_latfactors, size=(self.num_users, self.num_latfactors)
        )
        self.item_latfact_matrix = rng.normal(
            scale=1.0 / self.num_latfactors, size=(self.num_items, self.num_latfactors)
        )

        # Initialize the biases
        self.biase_users = np.zeros(self.num_users)
        self.biase_items = np.zeros(self.num_items)
        self.biase = np.mean(self.r_matrix[np.where(self.r_matrix != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.r_matrix[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.r_matrix[i, j] > 0
        ]

    def train(self):
        """
        Perform stochastic gradient descent for number of iterations.

        Training_process is a list of couples (id_iteration, mean_square_error)
        """

        #
        training_process = []
        for i in range(self.iterations):
            rng = np.random.RandomState(self.seed)
            rng.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, mse))

        return training_process

    def mse(self):
        """
        Compute the total mean square error.
        """

        rows_nonzero, columns_nonzero = self.r_matrix.nonzero()
        predicted = self.full_matrix()
        error = 0
        for r_nonzero, c_nonzero in zip(rows_nonzero, columns_nonzero):
            error += pow(
                self.r_matrix[r_nonzero, c_nonzero] - predicted[r_nonzero, c_nonzero], 2
            )
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic gradient descent.
        """
        for num_row, num_col, rat in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(num_row, num_col)
            error = rat - prediction

            # Update biases
            self.biase_users[num_row] += self.alpha * (
                error - self.beta * self.biase_users[num_row]
            )
            self.biase_items[num_col] += self.alpha * (
                error - self.beta * self.biase_items[num_col]
            )

            # Update user and item latent feature matrices
            self.user_latfact_matrix[num_row, :] += self.alpha * (
                error * self.item_latfact_matrix[num_col, :]
                - self.beta * self.user_latfact_matrix[num_row, :]
            )
            self.item_latfact_matrix[num_col, :] += self.alpha * (
                error * self.user_latfact_matrix[num_row, :]
                - self.beta * self.item_latfact_matrix[num_col, :]
            )

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j.
        """
        prediction = (
            self.biase
            + self.biase_users[i]
            + self.biase_items[j]
            + self.user_latfact_matrix[i, :].dot(self.item_latfact_matrix[j, :].T)
        )
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, users and items biases.
        """
        return (
            self.biase
            + self.biase_users[:, np.newaxis]
            + self.biase_items[np.newaxis:, ]
            + self.user_latfact_matrix.dot(self.item_latfact_matrix.T)
        )


R_MATRIX = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

M_F = MatrixFactorization(R_MATRIX)
tr_proc = M_F.train()

print(tr_proc)
