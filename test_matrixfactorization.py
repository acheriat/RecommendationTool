"""
Unit tests of the matrix factorization module.
"""

import numpy as np
import matrixfactorization


class TestMatrixFactorization:

    def test_get_rating(self):
        r_matrix = np.array([[5, 3, 0, 1], [4, 0, 0, 1]])
        mf = matrixfactorization.MatrixFactorization(r_matrix)
        assert mf.get_rating(1, 0) == 2.672935991777994

    def test_sgd(self):
        r_matrix = np.array([[5, 3, 0, 1], [4, 0, 0, 1]])
        mf = matrixfactorization.MatrixFactorization(r_matrix)
        mf.sgd()
        assert (mf.biase_users.tolist() == [0.007484458726407961, -0.035414091521230456] and
        mf.item_latfact_matrix.tolist() == [[-0.032641133006462326, -0.05555365174071124],
                                            [0.7854491962952435, 0.38496207636014385],
                                            [-0.23473719296747605, 0.27128002179298233],
                                            [-0.33370623672354505, -0.33482884637651444]])

    def test_full_matrix(self):
        r_matrix = np.array([[5, 3, 0, 1], [4, 0, 0, 1]])
        mf = matrixfactorization.MatrixFactorization(r_matrix)
        assert mf.full_matrix().tolist() == [[2.7790163718826255, 2.9695771323896034, 2.7229471856900074,
                                              2.758551918034064],
                                             [2.672935991777994, 3.347916011292537, 2.9305654916433834,
                                              2.5476323380768116]]

    def test_mes(self):
        r_matrix = np.array([[5, 3, 0, 1], [4, 0, 0, 1]])
        mf = matrixfactorization.MatrixFactorization(r_matrix)
        assert mf.mse() == 3.4903385812037913

    def test_train(self):
        r_matrix = np.array([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
        ])
        mf = matrixfactorization.MatrixFactorization(r_matrix)
        assert mf.train() == [(0, 5.468596914045281), (1, 4.954645307979005), (2, 4.300804333132003),
                              (3, 3.3569039342211373), (4, 2.126600667778168), (5, 1.2509019281100942),
                              (6, 0.8696031656138972), (7, 0.6818809684270286), (8, 0.5669838258059611),
                              (9, 0.48277705082712996), (10, 0.4003761329325773), (11, 0.3465686840777396),
                              (12, 0.2835041378952923), (13, 0.24552957914749302), (14, 0.21526948953431074),
                              (15, 0.18468907691426226), (16, 0.1628286226059519), (17, 0.13394758593152065),
                              (18, 0.11881052024819237), (19, 0.10858704529104617)]
