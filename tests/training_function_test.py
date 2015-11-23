from nose.tools import *
import networkx as nx
from supervised_random_walks.training_function import TrainingFunction

class TestTrainingFunction(object):
    def setup(self):
        self.func = TrainingFunction(C=[0,1,1,1])

    def teardown(self):
        "Teardown"

    def test_training_nodes(self):
        assert_equal(self.func.training_nodes(), [0,1,1,1])

    def test_initial_w(self):
        assert_equal(self.func.initial_w(), [])

    def test_feature_vector(self):
        assert_equal(self.func.feature_vector(), [[[]]])
