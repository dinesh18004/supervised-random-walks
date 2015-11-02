from nose.tools import *
from supervised_random_walks.square_loss import square_loss

def test_func():
    return 3-1
    
def square_loss_test():
    y = 1
    assert_equal(square_loss(y,test_func), 1)
