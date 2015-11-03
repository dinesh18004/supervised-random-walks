from nose.tools import *
from supervised_random_walks.square_loss import *

def test_diff():
    return 3-1

def square_loss_test_non_zero():
    y = 1
    assert_equal(square_loss(y, test_diff()), 1)

def square_loss_test_zero():
    y = 1
    assert_equal(square_loss(y, -1), 0)

def squaer_loss_der_test():
    y = 1
    assert_equal(square_loss_der(y, test_diff()), 2)
