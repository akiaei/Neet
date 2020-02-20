from abc import ABCMeta, abstractmethod
from neet.boolean import LogicNetwork
import neet
import networkx as nx
import numpy as np
#import Queue as Q

class TimeoutError(RuntimeError):
    """
    An error to signify that a network randomization algorithm failed to
    produce a valid network within the specified number of attempts.
    """
    pass

class AbstractConstraint(object, metaclass=ABCMeta):
    """
    An abstract class representing a constraint used for rejection testing.
    """
    @abstractmethod
    def satisfies(self, net):
        pass

class GenericConstraint(AbstractConstraint):
    def __init__(self, test):
        """
        A generic constraint constructable from a callable.
        """
        if not callable(test):
            raise(ValueError("generic constraint tests must be callable"))
        self.test = test

    def satisfies(self, net):
        return self.test(net)

AbstractConstraint.register(GenericConstraint)

class AbstractRandomizer(object, metaclass=ABCMeta):
    def __init__(self, network, constraints=list(), timeout=1000, **kwargs):
        """
        An abstract interface for all randomizers.
        """
        if isinstance(network, neet.Network):
            self.network = network
        elif isinstance(network, nx.DiGraph):
            self.__network = None
            self.__graph = network

        self.timeout = timeout
        self.constraints = constraints

    @property
    def network(self):
        return self.__net

    @network.setter
    def network(self, network):
        if not isinstance(network, neet.Network):
            raise(TypeError('network must be an instance of neet.Network'))
        self.__net = network
        self.__graph = self.__net.network_graph()

    @property
    def graph(self):
        return self.__graph

    @graph.setter
    def graph(self, graph):
        if not isinstance(graph, neet.Network):
            raise(TypeError('graph must be an instance of nx.DiGraph'))
        self.__net = None
        self.__graph = graph

    @property
    def constraints(self):
        return self.__constraints

    @constraints.setter
    def constraints(self, constraints):
        for i, constraint in enumerate(constraints):
            if isinstance(constraint, AbstractConstraint):
                continue
            elif callable(constraint):
                constraints[i] = GenericConstraint(constraint)
            else:
                raise(ValueError("constraints must be instances of AbstractConstraint or callable"))

        self.__constraints = constraints

    def add_constraint(self, constraint):
        if isinstance(constraint, AbstractConstraint):
            pass
        elif callable(constraint):
            constraint = GenericConstraint(constraint)
        else:
            raise(ValueError("constraints must be instances of AbstractConstraint or callable"))

        self.__constraints.append(constraint)

    def _check_constraints(self, net):
        """
        Check a network against the randomizer's constraints
        """
        for constraint in self.constraints:
            if not constraint.satisfies(net):
                return False
        return True

    def __iter__(self):
        """
        Generate an infinite list of random networks
        """
        while True:
            yield self.random()

    def random(self):
        """
        Attempt to generate a random network subject to the constraints
        """
        for _ in range(self.timeout):
            net = self._randomize()
            if self._check_constraints(net):
                return net
        raise(TimeoutError("failed to generate network that satisfies all constraints"))

    @abstractmethod
    def _randomize(self):
        """
        Generate a random network irrespective of the constraints
        """
        pass

class TopologyRandomizer(AbstractRandomizer):
    pass

class FixedStructure(TopologyRandomizer):
    """
    Generate the same topology - don't change anything
    """
    def __init__(self, network, constraints=list(), timeout=1000):
        super(FixedStructure, self).__init__(network, constraints, timeout)
        if not self._check_constraints(network):
            raise(ValueError("the provided network is inconsistent with the provided constraints"))

    def _randomize(self):
        return self.graph

AbstractRandomizer.register(FixedStructure)

class FixedMeanDegree(TopologyRandomizer):
    """
    Generate a topology with the same mean degree.
    """
    # Honestly, brilliant implementation here Doug. Elegant AF
    def _randomize(self):
        n = len(self.graph)
        edgeindices = np.random.choice(n*n, self.graph.size(), replace=False)

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(map(lambda i: divmod(i, n), edgeindices))
        return G

AbstractRandomizer.register(FixedMeanDegree)

# <W I P>
class FixedOutDegree(TopologyRandomizer):
    """
    Generate a topology with the exact same per-node out-degree.
    """
    #len(self.graph) returns number of nodes
    #self.graph.size() returns number of edges
    def _randomize(self):
        n = len(self.graph)
        outDegreeSequence = [sum(1 for _ in self.graph.successors(x)) for x in range(n)]
        # Returns a list containing the out-degree of each node in the graph.
        # outDegreeSequence[q] contains the number of outward-facing edges from
        # q to any other nodes

        edgeList = []
        for u in range(len(self.graph)):
            edgeList.extend([(u, x) for x in np.random.choice(np.delete(u, np.arange(n)), outDegreeSequence[u], replace=False)])
        # Basically, it goes through each edge u and creates a list of unique tuples
        # of the form (u, x) where x is any edge != u. 
        # "np.delete(u, np.arange(n))" returns a 1D array from 0 to n with the element u 
        # taken out.
        # The number of times this is done for each edge is determined by that edge's 
        # out-degree established above. That list of tuples is then concatenated with 
        # the rest of the edgeList

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edgeList)
        return G

AbstractRandomizer.register(FixedOutDegree)

class FixedInDegree(TopologyRandomizer):
    """
    Generate a topology with the exact same per-node in-degree.
    """
    #len(self.graph) returns number of nodes
    #self.graph.size() returns number of edges
    def _randomize(self):
        n = len(self.graph)
        inDegreeSequence = [sum(1 for _ in self.graph.predecessors(x)) for x in range(n)]
        # Returns a list containing the in-degree of each node in the graph.
        # inDegreeSequence[q] contains the number of incoming edges from any
        # other nodes to q

        edgeList = []
        for u in range(len(self.graph)):
            edgeList.extend([(x, u) for x in np.random.choice(np.delete(u, np.arange(n)), inDegreeSequence[u], replace=False)])
        # Basically, it goes through each edge u and creates a list of unique tuples
        # of the form (x, u) where x is any edge != u. 
        # "np.delete(u, np.arange(n))" returns a 1D array from 0 to n with the element u 
        # taken out.
        # The number of times this is done for each edge is determined by that edge's 
        # in-degree established above. That list of tuples is then concatenated with 
        # the rest of the edgeList

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edgeList)
        return G

AbstractRandomizer.register(FixedInDegree)
# </W I P>

class NetworkRandomizer(AbstractRandomizer):
    def __init__(self, network, topogen=None, constraints=list(), timeout=1000, p=0.5):
        super(NetworkRandomizer, self).__init__(network, constraints, timeout)
        if topogen is None:
            topogen = FixedStructure(network)
        self.topogen = topogen
        self.p = p

    def _randomize(self, topo=None):
        if topo is None:
            topo = self.topogen.random()

        table = []
        for node in np.sort(topo.nodes):
            predecessors = tuple(topo.predecessors(node))
            table.append((predecessors, self._random_function(len(predecessors))))
        return LogicNetwork(table)

    def _random_function(self, k):
        """
        Generate a random Boolean function with k inputs
        """
        p = self.p

        integer, decimal = divmod(2**k * p, 1)
        num_states = int(integer + np.random.choice(2, p=[1 - decimal, decimal]))
        state_idxs = np.random.choice(2**k, num_states, replace=False)

        return set('{0:0{1}b}'.format(idx, k) for idx in state_idxs)

AbstractRandomizer.register(NetworkRandomizer)

#class CustomRandomizer(AbstractRandomizer):
    #def __init__(self, network, topogen=None, constraints=list(), timeout=1000, p=0.5):
        

def testFID(network):

    nxl = len(network.graph)

    InNodeArray = np.sort([sum(1 for _ in network.graph.predecessors(x)) for x in range(nxl)])
    boolFlag = False
    for x in range(10):
        newFIDrandomizer = FixedInDegree(network)
        nxw = len(newFIDrandomizer.graph)
        InNodeArrayTwo = np.sort([sum(1 for _ in newFIDrandomizer.graph.predecessors(x)) for x in range(nxw)])
        if (InNodeArray == InNodeArrayTwo):
            print(InNodeArray, "\n", InNodeArrayTwo)
            boolFlag = True

        if boolFlag:
            print("F")
        else:
            print("!")
            #print(network.graph)
            #print(newFIDrandomizer.graph)
            print(InNodeArrayTwo)