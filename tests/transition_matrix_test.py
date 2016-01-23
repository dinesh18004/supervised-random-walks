from nose.tools import *
from supervised_random_walks.transition_matrix import *
from supervised_random_walks.edge_strength import *

class TestTransitonMatrix(object):
    def setup(self):
        self.stm = stochastic_transition_matrix(np.matrix([[0,1,2,2],
                                                           [1,0,2,2],
                                                           [2,2,0,1],
                                                           [2,2,1,0]]))

    def teardown(self):
        "Teardown"

    def test_stochastic_transition_matrix_size(self):
        assert_equal(self.stm.shape, (4,4))

    def test_stochastic_transition_matrix(self):
        v = self.stm == [[0, 0.2, 0.4, 0.4],
                         [0.2, 0, 0.4, 0.4],
                         [0.4, 0.4, 0, 0.2],
                         [0.4, 0.4, 0.2, 0]]
        assert_equal(v.all(), True)

    def test_stochastic_transition_matrix_returns_stochastic_matrix(self):
        row_sum = self.stm.sum(axis=1)
        v = row_sum == [[1],[1],[1],[1]]
        assert_equal(v.all(), True)

    def test_final_transition_matrix_sums_to_one_with_alpha(self):
        Q = final_transition_matrix(self.stm, alpha=0.3)
        row_sum = Q.sum(axis=1) + 0.3
        v = row_sum == [[1],[1],[1],[1]]
        assert_equal(v.all(),True)

    def test_final_transition_matrix_derivative(self):
        A = np.matrix([[0, 2, 2],
                       [2, 0, 2],
                       [2, 2, 0]])
        Q_prime = stochastic_transition_matrix(A)
        Q = final_transition_matrix(Q_prime, alpha=0.3)
        psi =  np.array([[
                         [[0,0],[1,1],[1,1]],
                         [[1,1],[0,0],[1,1]],
                         [[1,1],[1,1],[0,0]]
                        ]])
        dA = edge_strength_derivative(psi, A)
        #raise Exception(dA)
        #raise Exception(final_transition_matrix_derivative(Q, A, dA))
