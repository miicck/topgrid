from enum import Enum
from typing import Set


class TopologyPlot(Enum):
    X_AXIS_VALUES = 0,
    ALL_POINTS = 1,
    BULK_AND_EXTENDED_MAXIMA = 2,
    EXTENDED_MAXIMA_FAMILIES = 3,
    CRITICAL_TREE = 4,
    CLEAVED_CRITICAL_TREE = 5,
    CRITICAL_NETWORK = 6,
    STATIONARY_TREE = 7,
    CLEAVED_STATIONARY_TREE = 8,
    CONVEX_HULL = 9,
    SADDLE_LINKED_REGION_GRAPH = 10,
    REGION_PATHS = 11,
    STATIONARY_PATH_STATS = 12,
    LOCAL_MAXIMA_VALUES_HIST = 13,

    @staticmethod
    def all() -> Set['TopologyPlot']:
        return set(p for p in TopologyPlot)
