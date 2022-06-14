from enum import Enum

class TopologyInput:

    def __init__(self):

        self.plots = set()
        self.stationary_value_tolerance = 0.05
        self.network_cleaving_threshold = None
        self.convex_hull_depth = 0
        self.graph_ascent_method = GraphSteepestAscentMethod.ON_GRAPH
        self.nearest_neighbour_graph = None

class FunctionContractionMethod(Enum):
    MODULUS = 0,
    USE_SPECIFIC_INDEX = 1,
    MODULUS_OF_SPECIFIC_INDICES = 2,
    SUM = 3,
    SUM_OF_SPECIFIC_INDICES = 4


class FunctionTransformation(Enum):
    IDENTITY = 0,
    NEGATIVE = 1,
    ABSOLUTE = 2,
    NEGATIVE_ABSOLUTE = 3,
    POSITIVE_COMPONENT = 4,
    NEGATIVE_COMPONENT = 5,
    ISOSURFACE = 6,


class GraphSteepestAscentMethod(Enum):
    ON_GRAPH = 0,
    OFF_GRAPH = 1,
    MONTE_CARLO = 2
