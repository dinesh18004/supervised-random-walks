from nose.tools import *
import networkx as nx
import supervised_random_walks

class TestSupervisedRandomWalks(object):
    @classmethod
    def setup_class(klass):
        """This method is run once for each class before any tests are run"""

    @classmethod
    def teardown_class(klass):
        """This method is run once for each class _after_ all tests are run"""

    def setUp(self):
        self.graph = nx.fast_gnp_random_graph(200,.8)

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    def test_initialized_with_(self):
        print(self.graph)
        assert_not_equal(True, False)
