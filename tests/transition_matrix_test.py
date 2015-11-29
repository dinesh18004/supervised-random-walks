from nose.tools import *
from supervised_random_walks.transition_matrix import *

class TestTransitonMatrix(object):
    def setup(self):
        self.stm = stochastic_transition_matrix(np.matrix([[1,2,3,4],[1,2,3,4]]))

    def teardown(self):
        "Teardown"

    def test_stochastic_transition_matrix_size(self):
        assert_equal(self.stm.shape, (2,4))

    def test_stochastic_transition_matrix(self):
        v = self.stm == [[0.1,0.2,0.3,0.4], [0.1,0.2,0.3,0.4]]
        assert_equal(self.stm.all(), True)

    def test_stochastic_transition_matrix_returns_stochastic_matrix(self):
        row_sum = self.stm.sum(axis=1)
        v = row_sum == [[1],[1]]
        assert_equal(v.all(), True)

    def test_final_transition_matrix(self):
        Q = final_transition_matrix(self.stm)
        row_sum = Q.sum(axis=1)
        v = row_sum == [[0.8],[0.8]]
        assert_equal(v.all(),True)
