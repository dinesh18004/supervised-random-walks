from nose.tools import *
from supervised_random_walks.transition_matrix import *

class TestTransitonMatrix(object):
    def setup(self):
        self.tm = transition_matrix(np.matrix([[1,2,3,4],[1,2,3,4]]))

    def teardown(self):
        "Teardown"

    def test_transition_matrix_size(self):
        assert_equal(self.tm.shape, (2,4))

    def test_transition_matrix(self):
        v = self.tm == [[0.1,0.2,0.3,0.4], [0.1,0.2,0.3,0.4]]
        assert_equal(self.tm.all(), True)

    def test_transition_matrix_right_stochastic_matrix(self):
        row_sum = self.tm.sum(axis=1)
        v = row_sum == [[1],[1]]
        assert_equal(v.all(), True)
