from nose.tools import *
from supervised_random_walks.transition_matrix import *


def test_transition_matrix_size():
    tm = transition_matrix(np.matrix([[1,2,3,4],[1,2,3,4]]))
    assert_equal(tm.shape, (2,4))
