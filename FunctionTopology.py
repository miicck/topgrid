import os
import random
from time import time
from collections import defaultdict

# Math
import numpy as np
from scipy.spatial import Delaunay, KDTree

# Network algorithms
import networkx
from networkx.algorithms.tree.mst import maximum_spanning_tree, minimum_spanning_tree
from networkx.algorithms.shortest_paths.generic import shortest_path
from networkx.algorithms.components.connected import connected_components

# Imports for typing only
from typing import Tuple, List, Dict, Iterator, Sequence, Set, Optional

# Import topology input section object for typing and to copy default values from
from topgrid.TopologyInput import TopologyInput, GraphSteepestAscentMethod
from topgrid.TopologyPlot import TopologyPlot


class FunctionTopology:

    def __init__(self,
                 points: Sequence[Sequence[float]],
                 function: Sequence[float],
                 gradients: Sequence[float] = None,
                 settings: TopologyInput = None):
        """
        Create a function topology for the given
        function defined on the given set of points.

        Parameters
        ----------
        points: Sequence[Sequence[float]]
            Points in R^N at which the function is defined; points[i] should return the i^th point.
        function: Sequence[float]
            For each point, the value of the function at that point.
        """

        # Will contain already-calculated things for retrieval
        self._stored_properties = dict()
        self._analytic_gradients = None if gradients is None else np.array(gradients)

        # Convert points/function to numpy arrays and assert they are compatible shapes
        self.points: np.ndarray = np.array(points)
        self.function: np.ndarray = np.array(function)
        assert len(self.function.shape) == 1
        assert len(self.points.shape) == 2
        assert self.points.shape[0] == self.function.shape[0]

        if self._analytic_gradients is not None:
            assert self._analytic_gradients.shape == self.points.shape

        # Use default settings if no settings provided
        self.settings = settings or TopologyInput()

    ##############
    # PROPERTIES #
    ##############

    @property
    def points(self) -> np.ndarray:
        """
        The points at which the function whose
        topology we are analysing is evaluated.
        """
        return self._points

    @points.setter
    def points(self, val: np.ndarray):
        self._points = val
        self.clear_stored_properties()  # Points changed => we need to re-evaluate things

    @property
    def function(self) -> np.ndarray:
        """
        The function whose topology we're analysing.
        """
        return self._function

    @function.setter
    def function(self, func: np.ndarray):
        self._function = np.array(func)
        self.min_function = min(func)
        self.max_function = max(func)
        self.clear_stored_properties()  # Function changed => we need to re-evaluate things

    @property
    def print_method(self) -> callable:
        return TimeAndStore.print_method

    @print_method.setter
    def print_method(self, value):
        TimeAndStore.print_method = value

    @property
    def stored_properties_count(self):
        return len(self._stored_properties)

    def clear_stored_properties(self):
        self._stored_properties.clear()

    ########################
    # READ-ONLY PROPERTIES #
    ########################

    @property
    def dimensions(self) -> int:
        return self.points.shape[1]

    @property
    def triangulation(self) -> Delaunay:
        with TimeAndStore(self._stored_properties, "Evaluating triangulation") as stored:
            if stored.value is not None:
                return stored.value

            self.print(f"Evaluating topology of function in range "
                       f"[{self.min_function:.3}, {self.max_function:.3}] "
                       f"defined on {len(self.points)} points...")

            stored.value = Delaunay(self.points, incremental=False)
            return stored.value

    @property
    def graph(self) -> networkx.Graph:
        """
        A graph over input points, with edges connecting neighbouring points.
        """
        with TimeAndStore(self._stored_properties, "Constructing graph") as stored:
            if stored.value is not None:
                return stored.value

            if self.dimensions == 1:
                graph = networkx.Graph()
                indices = list(range(len(self.points)))
                indices.sort(key=lambda i: self.points[i][0])
                graph.add_edges_from((indices[n - 1], indices[n]) for n in range(1, len(indices)))
                stored.value = graph
                return graph

            if self.settings.nearest_neighbour_graph is not None:
                graph = networkx.Graph()
                for i in range(len(self.points)):
                    dists, index = self.kd_tree.query(self.points[i], k=self.settings.nearest_neighbour_graph + 1)
                    for j in index:
                        if i != j:
                            graph.add_edge(i, j)
                stored.value = graph
                return graph

            # All vertices from the same simplex, or coplanar vertices are edges on the graph
            graph = networkx.Graph()
            graph.add_edges_from((i, j) for simplex in self.triangulation.simplices
                                 for i in simplex for j in simplex if i < j)
            graph.add_edges_from((i, j) for i, facet, j in self.triangulation.coplanar)

            stored.value = graph
            return graph

    @property
    def gradient(self) -> np.ndarray:
        """
        The gradient at each point, such that gradient[i] is a d-dimensional vector.
        """
        with TimeAndStore(self._stored_properties, "Evaluating gradients") as stored:
            if stored.value is not None:
                return stored.value

            if self._analytic_gradients is not None:
                # Use analytic gradients if they are provided
                stored.value = self._analytic_gradients
                return stored.value

            else:
                # Fallback to numerical gradients
                stored.value = self.numerical_gradients
                return stored.value

    @property
    def numerical_gradients(self) -> np.ndarray:
        with TimeAndStore(self._stored_properties, "Evaluating numerical gradients") as stored:
            if stored.value is not None:
                return stored.value

            def min_residual_gradient(i):
                deltas = [self.points[n] - self.points[i] for n in self.graph[i]]
                delta_functions = [self.function[n] - self.function[i] for n in self.graph[i]]
                matrix = sum(np.outer(d, d) for d in deltas)
                vector = sum(d * fd for d, fd in zip(deltas, delta_functions))
                return np.linalg.inv(matrix) @ vector

            stored.value = [min_residual_gradient(i) for i in range(len(self.points))]
            return stored.value

    @property
    def kd_tree(self) -> KDTree:
        """
        The KD Tree of my points.
        """
        with TimeAndStore(self._stored_properties, "Evaluating KD tree") as stored:
            if stored.value is not None:
                return stored.value

            stored.value = KDTree(self.points)
            return stored.value

    @property
    def average_function_edge_attributes_set(self) -> bool:

        with TimeAndStore(self._stored_properties, "Setting average function edge attributes") as stored:
            if stored.value is not None:
                return stored.value

            networkx.set_edge_attributes(
                self.graph,
                {tuple(e): (self.function[e[0]] + self.function[e[1]]) / 2.0 for e in self.graph.edges},
                "average function value")

            stored.value = True
            return True

    @property
    def abs_gradient_edge_attributes_set(self) -> bool:

        with TimeAndStore(self._stored_properties, "Setting absolute gradient edge attributes") as stored:
            if stored.value is not None:
                return stored.value

            def abs_gradient(e):
                delta = self.function[e[0]] - self.function[e[1]]
                delta /= np.linalg.norm(self.points[e[0]] - self.points[e[1]])
                return abs(delta)

            networkx.set_edge_attributes(
                self.graph,
                {tuple(e): abs_gradient(e) for e in self.graph.edges},
                "abs gradient")

            stored.value = True
            return True

    @property
    def convex_hull(self) -> Set[int]:

        with TimeAndStore(self._stored_properties, "Evaluating convex hull") as stored:
            if stored.value is not None:
                return stored.value

            if self.dimensions == 1:
                hull = set()
                hull.add(max(range(len(self.points)), key=lambda i: self.points[i][0]))
                hull.add(min(range(len(self.points)), key=lambda i: self.points[i][0]))
                stored.value = hull
                return hull

            hull = set()
            for face in self.triangulation.convex_hull:
                for index in face:
                    hull.add(index)

            for i in range(self.settings.convex_hull_depth - 1):

                new_hull = set(hull)
                for index in hull:
                    for neighbour in self.graph[index]:
                        new_hull.add(neighbour)
                hull = new_hull

            stored.value = hull
            return hull

    @property
    def climb_graph(self) -> networkx.DiGraph:
        """
        A directed spanning tree over the whole network that describes
        the path from any given point to it's associated maximum.
        """
        with TimeAndStore(self._stored_properties, "Evaluating climb graph") as stored:
            if stored.value is not None:
                return stored.value

            climb_graph = networkx.DiGraph()

            for index in range(len(self.points)):

                climb = []
                for i in self.climb(index, use_climb_graph=False):

                    climb.append(i)

                    # Climb until we reach a graphed node
                    if i in climb_graph:
                        break

                if len(climb) == 1:
                    climb_graph.add_node(climb[0])

                for i in range(1, len(climb)):
                    a = climb[i - 1]
                    b = climb[i]
                    climb_graph.add_edge(a, b, weight=self.edge_weight(a, b))

            stored.value = climb_graph
            return climb_graph

    @property
    def local_maxima(self) -> Set[int]:
        """
        Points that are local maxima on the graph.
        """
        with TimeAndStore(self._stored_properties, "Identifying local maxima") as stored:
            if stored.value is not None:
                return stored.value
            stored.value = {i for i in self.climb_graph if len(self.climb_graph[i]) == 0}
            return stored.value

    @property
    def bulk_maxima(self) -> Set[int]:
        with TimeAndStore(self._stored_properties, "Identifying bulk maxima") as stored:
            if stored.value is not None:
                return stored.value
            stored.value = self.local_maxima - self.convex_hull
            return stored.value

    #################
    # CRITICAL TREE #
    #################

    @property
    def max_spanning_tree(self) -> networkx.Graph:
        """
        A maximum spanning tree connecting all input points
        (maximum with respect to the values of the function on the graph).
        """
        with TimeAndStore(self._stored_properties, "Evaluating maximum spanning tree") as stored:
            if stored.value is not None:
                return stored.value
            if not self.average_function_edge_attributes_set:
                raise Exception("Edge attributes not set correctly")
            stored.value = maximum_spanning_tree(self.graph, weight="average function value")
            return stored.value

    @property
    def critical_tree(self) -> networkx.Graph:
        """
        A maximum spanning tree that connects all local maxima.
        This is essentially the maximum spanning tree of the whole
        graph, but pruned until it can't be pruned any more without
        removing a local maxima.
        """
        with TimeAndStore(self._stored_properties, "Evaluating critical tree") as stored:
            if stored.value is not None:
                return stored.value
            stored.value = FunctionTopology.pruned_tree(self.max_spanning_tree, self.bulk_maxima)
            return stored.value

    @property
    def critical_network(self) -> networkx.Graph:
        with TimeAndStore(self._stored_properties, "Evaluating critical network") as stored:
            if stored.value is not None:
                return stored.value
            stored.value = self.fill_critical_tree_cycles(self.critical_tree, set(self.region_centres.values()))
            return stored.value

    @property
    def cleaved_critical_tree(self) -> Tuple[Tuple[networkx.Graph], Dict]:
        with TimeAndStore(self._stored_properties, "Cleaving critical tree") as stored:
            if stored.value is not None:
                return stored.value
            stored.value = self.cluster_tree_by_flatness(self.critical_tree, self.bulk_maxima, self.function)
            return stored.value

    ###################
    # STATIONARY TREE #
    ###################

    @property
    def min_deviation_tree(self) -> networkx.Graph:
        """
        The minimum spanning tree of the graph when the edge weights are
        set to the absolute function gradient along that edge.
        """
        with TimeAndStore(self._stored_properties, "Generating minimum deviation tree") as stored:
            if stored.value is not None:
                return stored.value
            if not self.abs_gradient_edge_attributes_set:
                raise Exception("Edge attributes not set correctly")
            stored.value = minimum_spanning_tree(self.graph, weight="abs gradient")
            return stored.value

    @property
    def stationary_tree(self) -> networkx.Graph:
        """
        The minimum deviation tree, pruned until it only spans the local maxima.
        """
        with TimeAndStore(self._stored_properties, "Generating stationary tree") as stored:
            if stored.value is not None:
                return stored.value
            stored.value = FunctionTopology.pruned_tree(self.min_deviation_tree, self.bulk_maxima)
            return stored.value

    @property
    def cleaved_stationary_tree(self) -> Tuple[Tuple[networkx.Graph], Dict]:
        with TimeAndStore(self._stored_properties, "Cleaving stationary tree") as stored:
            if stored.value is not None:
                return stored.value
            stored.value = self.cluster_tree_by_flatness(self.stationary_tree, self.bulk_maxima, self.function)
            return stored.value

    ###################
    # EXTENDED MAXIMA #
    ###################

    @property
    def extended_maxima_families(self) -> Dict[int, Set[int]]:
        """
        Returns
        -------
        The disconnected sets of points resulting from performing a flood
        fill from each local maximum in the bulk, allowing only moves where the
        function does not vary too strongly. This essentially extends and merges
        local maxima into maxima families.
        """

        with TimeAndStore(self._stored_properties, "Generated extended maxima families") as stored:
            if stored.value is not None:
                return stored.value

            def deviation(i_from, i_to):
                f_from = self.function[i_from]
                f_to = self.function[i_to]
                return abs(f_from - f_to) / (f_from - self.min_function)

            floods: Dict[int, Set[int]] = dict()
            for lm in self.bulk_maxima:
                to_expand = {lm}
                flood = set()

                while len(to_expand) > 0:
                    expanding = to_expand.pop()
                    flood.add(expanding)

                    for n in self.graph[expanding]:
                        if n in to_expand or n in flood:
                            continue  # Already discovered
                        if deviation(lm, n) > self.settings.stationary_value_tolerance:
                            continue  # Too much deviation from initial maxima
                        to_expand.add(n)

                floods[lm] = flood

            def merge_floods():
                for i in floods:
                    for j in floods:
                        if i != j and len(floods[i].intersection(floods[j])) > 0:
                            floods[i].update(floods[j])
                            floods.pop(j)
                            return True
                return False

            # Merge all overlapping local-maxima floods
            while merge_floods():
                pass

            # Create a map of family id to the set of points in that family
            stored.value = {i + 1: floods[x] for i, x in enumerate(floods)}
            return stored.value

    @property
    def extended_maxima(self) -> Set[int]:
        """
        Returns
        -------
        The union of all maxima families.
        """
        with TimeAndStore(self._stored_properties, "Identifying extended maxima") as stored:
            if stored.value is not None:
                return stored.value
            stored.value = set().union(*(self.extended_maxima_families[i] for i in self.extended_maxima_families))
            return stored.value

    ###########
    # REGIONS #
    ###########

    @property
    def regions(self) -> List[int]:
        """
        For each point, the id of the maxima family whose
        basin of attraction it belongs to.
        """
        with TimeAndStore(self._stored_properties, "Identifying regions") as stored:
            if stored.value is not None:
                return stored.value

            # Region -1 => unassigned
            regions = [-1] * len(self.points)

            # Assign extended maxima to their family
            for family in self.extended_maxima_families:
                for em in self.extended_maxima_families[family]:
                    regions[em] = family

            def shortcut_possible_on_graph(j):
                return regions[j] >= 0

            def shortcut_possible_off_graph(j):
                return regions[j] >= 0 and all(regions[n] == regions[j] for n in self.graph[j])

            def shortcut_possible_monte_carlo(j):
                return False

            climb_method = {
                GraphSteepestAscentMethod.ON_GRAPH: lambda i: self.climb(i),
                GraphSteepestAscentMethod.OFF_GRAPH: self.climb_off_graph,
                GraphSteepestAscentMethod.MONTE_CARLO: self.climb_monte_carlo,
            }[self.settings.graph_ascent_method]

            shortcut_possible = {
                GraphSteepestAscentMethod.ON_GRAPH: shortcut_possible_on_graph,
                GraphSteepestAscentMethod.OFF_GRAPH: shortcut_possible_off_graph,
                GraphSteepestAscentMethod.MONTE_CARLO: shortcut_possible_monte_carlo
            }[self.settings.graph_ascent_method]

            # Assign other points according to their basin of attraction
            for i in range(len(self.points)):

                if regions[i] >= 0:
                    continue  # Already assigned

                climb = []
                for j in climb_method(i):
                    climb.append(j)

                    if shortcut_possible(j):
                        # We can shortcut the
                        # assignment to the region of j
                        for k in climb:
                            regions[k] = regions[j]
                        break

                if regions[i] >= 0:
                    continue  # Assigned successfully (via shortcut)

                # If the climb path ends on the convex hull
                # then assign the path to the convex hull
                if climb[-1] in self.convex_hull:
                    for k in climb:
                        regions[k] = 0
                    continue

                # Assign the region of i to the region at the top of the climb
                regions[i] = regions[climb[-1]]

                if regions[i] < 0:
                    raise Exception(f"Failed to generate a region for point {i}")

            stored.value = regions
            return stored.value

    @property
    def region_count(self) -> int:
        """
        The number of regions in the topology.
        """
        with TimeAndStore(self._stored_properties, "Evaluating number of regions") as stored:
            if stored.value is not None:
                return stored.value
            stored.value = len(set(self.regions))
            return stored.value

    @property
    def region_centres(self) -> Dict[int, int]:
        """
        A dictionary mapping region ids to indices
        considered to be the centre of that region
        """
        with TimeAndStore(self._stored_properties, "Evaluating region centres") as stored:
            if stored.value is not None:
                return stored.value

            # A region centre is the maximum function value in that region
            def max_index(region):
                return max((i for i in self.local_maxima if self.regions[i] == region),
                           key=lambda i: self.function[i])

            stored.value = {r: max_index(r) for r in set(self.regions)}
            return stored.value

    @property
    def region_paths(self) -> Dict[Tuple[int, int], List[int]]:
        """
        A dictionary mapping a pair of neighbouring regions
        to the maximum-function (critical) path between them.
        The paths are indexed by a pair of region ids in ascending order.
        """
        with TimeAndStore(self._stored_properties, "Evaluating region paths") as stored:
            if stored.value is not None:
                return stored.value

            region_paths = dict()
            for i, i_centre in self.region_centres.items():
                for j, j_centre in self.region_centres.items():
                    if i >= j:
                        continue

                    if i_centre not in self.critical_network:
                        assert i_centre in self.convex_hull
                        continue

                    if j_centre not in self.critical_network:
                        assert j_centre in self.convex_hull
                        continue

                    path = shortest_path(self.critical_network, source=i_centre, target=j_centre)

                    if len({self.regions[p] for p in path}) != 2:
                        continue  # Region paths should only pass through two regions
                    region_paths[(i, j)] = path

            stored.value = region_paths
            return stored.value

    @property
    def region_boundaries(self) -> Dict[Tuple[int, int], Set[int]]:
        """
        Returns
        -------
        boundaries:
            A map from a pair of regions (i, j) to the set of points in
            region i that are on the boundary with region j.
        """
        with TimeAndStore(self._stored_properties, "Evaluating region boundaries") as stored:
            if stored.value is not None:
                return stored.value

            region_boundaries: Dict[Tuple[int, int], Set[int]] = defaultdict(lambda: set())
            for i, ri in enumerate(self.regions):
                for n in self.graph[i]:
                    rn = self.regions[n]
                    if rn != ri:
                        region_boundaries[(ri, rn)].add(i)

            stored.value = region_boundaries
            return stored.value

    @property
    def regions_connected_by_saddle_point(self) -> Dict[int, Set[int]]:
        """
        Returns
        -------
        critical_links:
            A map from a region to the set of regions linked to it via a saddle point.
        """
        with TimeAndStore(self._stored_properties, "Evaluating saddle-point-linked regions") as stored:
            if stored.value is not None:
                return stored.value

            links: Dict[int, Set[int]] = defaultdict(lambda: set())
            for key in self.region_boundaries:

                boundary = self.region_boundaries[key]
                critical_point = max(boundary, key=lambda p: self.function[p])
                bordering = set(self.regions[n] for n in self.graph[critical_point])

                if len(bordering) > 2:
                    continue  # Not critically linked

                # Add the critical link both ways
                links[key[0]].add(key[1])
                links[key[1]].add(key[0])

            stored.value = links
            return stored.value

    @property
    def saddle_linked_region_graph(self) -> networkx.Graph:
        """
        Returns
        -------
        graph:
            The graph with edges between saddle-point-linked region centres.
        """
        # Work out the boundaries between regions
        edges = []
        for ri, ci in self.region_centres.items():
            for rj, cj in self.region_centres.items():

                if ci >= cj:
                    continue  # Avoid double counting edges

                if ri == rj:
                    continue  # Don't connect points in same region

                if ri in self.regions_connected_by_saddle_point[rj]:
                    edges.append([ci, cj])

        filled_tree = networkx.Graph()
        filled_tree.add_edges_from(edges)
        return filled_tree

    ###########
    # METHODS #
    ###########

    def region_charges(self, weights: Sequence[float]) -> Dict[int, float]:
        """
        Evaluate the sum of the given weights array for each distinct region.

        Parameters
        ----------
        weights: Sequence[float]
            A weight for each point, to sum over each region.

        Returns
        -------
        region_charges: Dict[int, float]
            For each region id (int), the summed weight over that region (float).
        """
        assert len(weights) == len(self.points)

        with TimeAndStore(self._stored_properties, "Evaluating region charges") as stored:
            if stored.value is not None:
                return stored.value

            result = dict()
            for i, weight in enumerate(weights):
                weight *= self.function[i]
                r = self.regions[i]
                if r in result:
                    result[r] += weight
                else:
                    result[r] = weight

            stored.value = result
            return stored.value

    def edge_weight(self, i: int, j: int) -> float:
        """
        Evaluate the weight that should be assigned
        to an edge connecting indices i and j.

        Parameters
        ----------
            i: int
                Index of first node in the graph.
            j: int
                Index of second node in the graph.
        """
        scaled_f = (self.function[i] + self.function[j]) * 0.5
        scaled_f = (scaled_f - self.min_function) / (self.max_function - self.min_function)
        return scaled_f

    def nearest_point(self, location) -> int:
        """
        Find the nearest point on the graph to the given location.
        """
        dists, index = self.kd_tree.query(location, k=1)
        if index == len(self.points):
            raise Exception(f"Could not identify nearest neighbour of point {location}")
        return index

    def climb_off_graph(self, index: int) -> Iterator[int]:
        """
        Generate a set of indices along which self.function increases,
        according to a steepest-gradient path that is allowed to go
        off-graph.
        Parameters
        ----------
        index:
            The index to start climbing from (will be yielded immediately).
        Yields
        ------
            index: int
                The next index along the path.
        """

        def steepest_ascent_direction(i):
            # Fallback is just steepest-ascent direction
            for j in self.climb(i):
                if j != i:
                    return self.points[j] - self.points[i]

        # Start at the point at the given index
        x = np.array(self.points[index])
        visited = set()
        iteration = 0

        while True:

            # Work out the nearest grid point to x
            i = self.nearest_point(x)

            iteration += 1
            if iteration % 1000 == 0:
                self.print(f"Reached off-graph climb iteration {iteration} for point {index}. Got to point {i} x = {x}")

            if i not in visited:
                yield i
                visited.add(i)

            if i in self.local_maxima:
                return  # We've reached the top

            # Use the nearest-neighbour gradient
            g = self.gradient[i]
            g_norm = np.linalg.norm(g)

            if g_norm < 1e-5:
                g = steepest_ascent_direction(i)
                g_norm = np.linalg.norm(g)

            # Use the minimum edge size as our step size
            s = min(np.linalg.norm(self.points[j] - self.points[i]) for j in self.graph[i])

            # Make a step
            step = s * g / g_norm
            x += step

    def climb(self, index: int, use_climb_graph: bool = True) -> Iterator[int]:
        """
        Generate a set of indices along which self.function
        increases, starting at the given index. The indices
        will be generated by making moves along the path with
        the largest gradient in the self.neighbours network.

        Parameters
        ----------
            index: int
                The index to start climbing from (will be yielded immediately).
            use_climb_graph: bool
                True if we should use the already-constructed
                climb_graph to speed up generation of the path.

        Yields
        ------
            index: int
                The next index along the path.
        """
        yield index

        if use_climb_graph:
            # Use the already-constructed climb graph
            i = index

            while len(self.climb_graph[i]) > 0:
                if len(self.climb_graph[i]) > 1:
                    raise Exception("Climb graph is not one-to-one!")
                for j in self.climb_graph[i]:
                    i = j
                yield i

            return

        visited = {index}

        # Climb self.graph manually
        while True:

            # Evaluate the function and coordinates
            # at the current index
            f0 = self.function[index]
            p0 = self.points[index]

            max_index = index
            max_dfdx = -float("inf")

            # Search neighbours of the current
            # point for the largest gradient
            for n in self.graph[index]:

                if n == index:
                    # Don't allow self-moves
                    continue

                fn = self.function[n]
                if fn < f0:
                    # Don't allow downhill moves
                    continue

                if n in visited:
                    # Don't allow moves to already-visited points
                    # this allows the climber to explore regions
                    # where the function is flat, without getting stuck
                    # in an infinite loop.
                    continue

                dfdx = fn - f0
                dfdx /= np.linalg.norm(self.points[n] - p0)

                if dfdx > max_dfdx:
                    # Found a larger-gradient move
                    max_index = n
                    max_dfdx = dfdx

            if max_index == index:
                # No move made => we're at a maximum
                return

            # Make move along the maximum gradient path
            index = max_index
            visited.add(index)
            yield index

    def climb_monte_carlo(self, index: int) -> Iterator[int]:
        """
        Generate a set of indices along which self.function increases,
        according to a probability along each upward direction, so that
        the average move aligns with the gradient.


        Parameters
        ----------
        index:
            The index to start climbing from (will be yielded immediately).
        Yields
        ------
            index: int
                The next index along the path.
        """
        raise NotImplementedError()

    def fill_critical_tree_cycles(self, tree: networkx.Graph, nodes: Set[int]) -> networkx.Graph:
        """
        Adds missing critical paths to the critical tree, by attempting
        to add critical paths through subgraphs.
        Parameters
        ----------
        tree: networkx.Graph
            The tree to fill cycles in.

        Returns
        -------
        filled_tree: networkx.Graph
            The graph with cycles in the tree filled.
        """
        new_edges = []

        nodes = nodes.intersection(tree.nodes)
        pairs = [[n1, n2] for n1 in nodes for n2 in nodes if n1 < n2]

        subgraphs = dict()
        subtrees = dict()
        subpaths = dict()

        for source, target in pairs:

            # Check if already critically linked on tree
            path = shortest_path(tree, source=source, target=target)
            if len(set(self.regions[p] for p in path)) <= 2:
                self.print(f"Points {source} and {target} already critically linked")
                continue

            # Get regions to path between
            r_source = self.regions[source]
            r_target = self.regions[target]

            if r_source != r_target and (r_source, r_target) not in self.region_boundaries:
                continue  # These regions do not border one another

            # Check if subgraph/tree has already been generated fro this pair of regions
            key = (r_source, r_target) if r_source < r_target else (r_target, r_source)
            if key in subgraphs:
                subgraph = subgraphs[key]
                subtree = subtrees[key]
            else:
                self.print(f"Generating subgraph for regions {r_source} and {r_target}...")
                subnodes = (n for n in self.graph if self.regions[n] in {r_source, r_target})
                subgraph = subgraphs[key] = networkx.subgraph(self.graph, subnodes)
                subtree = subtrees[key] = maximum_spanning_tree(subgraph, weight="average function value")

            # Check if subpath has already been generated
            key = (source, target)
            if key in subpaths:
                subpath = subpaths[key]
            else:
                self.print(f"Generating subpath for points {source} and {target}...")
                subpath = subpaths[key] = shortest_path(subtree, source=source, target=target)

            def on_edge(i):
                return len(set(self.regions[j] for j in self.graph[i]) - {r_source, r_target}) > 0

            # A critical path touches the edge of r_source \cup r_target => it wants to go through
            # another region => r_source and r_target are not critically linked
            if any(on_edge(p) for p in subpath):
                self.print(f"Points {source} and {target} are not critically linked")
                continue

            for i in range(1, len(subpath)):
                new_edges.append((subpath[i - 1], subpath[i]))

        tree = tree.copy()
        tree.add_edges_from(new_edges)
        return tree

    ##################
    # SAVING/LOADING #
    ##################

    def save_to_disk(self, filename=None) -> None:
        import pickle
        filename = filename or self.settings.topology_save_file
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filename=None) -> 'FunctionTopology':
        import pickle
        filename = filename or TopologyInput().topology_save_file
        with open(filename, "rb") as f:
            return pickle.load(f)

    ##################
    # STATIC METHODS #
    ##################

    @staticmethod
    def pruned_tree(tree: networkx.Graph, nodes_to_keep: Set[int]) -> networkx.Graph:
        """
        Given an input tree, return a copy that has been maximally pruned whilst still spanning the given nodes.
        Parameters
        ----------
        tree: networkx.Graph
            The tree to prune.
        nodes_to_keep: Set[int]
            The nodes that must survive in the pruned tree.

        Returns
        -------
        pruned_tree: networkx.Graph
            The pruned tree.
        """
        tree = tree.copy()

        # Start at leaf nodes that are not local maxima
        to_expand = {n for n in tree.nodes if len(tree[n]) == 1 if n not in nodes_to_keep}
        to_remove = set()

        while len(to_expand) > 0:

            # Remove-and-expand the next leaf node
            expanding = to_expand.pop()
            to_remove.add(expanding)

            for n in tree[expanding]:

                if n in nodes_to_keep:
                    continue  # Don't expand to local maxima

                if n in to_remove:
                    continue  # Don't expand to already-removed nodes

                surviving_neighbours = len([j for j in tree[n] if j not in to_remove])
                if surviving_neighbours > 1:
                    continue  # Don't expand to non-leaf nodes

                to_expand.add(n)

        # Remove all nodes from graph at once (faster than one-by-one)
        tree.remove_nodes_from(to_remove)
        return tree

    @staticmethod
    def graph_has_cycles(graph: networkx.Graph) -> bool:
        try:
            networkx.find_cycle(graph)
            return True
        except networkx.NetworkXNoCycle:
            return False

    @staticmethod
    def critical_paths(tree: networkx.Graph, target_nodes: Set[int]) -> Dict[Tuple[int, int], List[int]]:
        """
        The paths between pairs of nodes on the given tree that
        do not pass through other nodes in the set.
        """

        if FunctionTopology.graph_has_cycles(tree):
            raise Exception("Can't find critical paths on graph with cycles!")

        paths = dict()
        for i in target_nodes:

            # Start paths along every branch from i
            paths_to_extend = [[i, n] for n in tree[i]]
            completed_paths = []

            while len(paths_to_extend) > 0:

                for path_i, path in enumerate(paths_to_extend):

                    # Check if this path has reached a target node, if so its complete
                    if path[-1] in target_nodes:
                        paths_to_extend.pop(path_i)
                        completed_paths.append(path)
                        break

                    # Find new branches from the end of the path
                    branches = [n for n in tree[path[-1]] if n != path[-2]]

                    # Nowhere to go => we've hit a leaf
                    if len(branches) == 0:
                        paths_to_extend.pop(path_i)
                        assert len(tree[path[-1]]) == 1  # Double check it is a leaf
                        break

                    # Only one way to go => just extend the path
                    if len(branches) == 1:
                        path.append(branches[0])
                        continue

                    # Create a new set of paths from the branches
                    new_paths = [path + [b] for b in branches]
                    paths_to_extend[path_i] = new_paths[0]  # Overwrite path with first branch
                    paths_to_extend.extend(new_paths[1:])  # Add all other branches to paths
                    break

            # Save paths indexed by start/end point
            for path in completed_paths:
                j = path[-1]
                paths[(i, j)] = path

        # Double checks for results
        for i, j in paths:
            assert (j, i) in paths
            forward_path = paths[(i, j)]
            backward_path = paths[(j, i)]
            assert forward_path == list(reversed(backward_path))
            assert forward_path[0] in target_nodes
            assert forward_path[-1] in target_nodes
            assert all(n not in target_nodes for n in forward_path[1:-1])

        # Only return forward paths (lower index -> higher index)
        return {key: paths[key] for key in paths if key[0] < key[1]}

    def cluster_tree_by_flatness(self, tree: networkx.Graph, nodes_to_cluster: Set[int], function: np.ndarray) -> \
            Tuple[Tuple[networkx.Graph], Dict]:
        """
        Given a tree containing the given nodes, this method will create subtrees by removing
        any paths on the input tree along which the function is not "flat" enough. This essentially
        clusters the given nodes into connected subsets for which the function does not vary strongly.
        Parameters
        ----------
        tree:
            The tree to create subtrees from
        nodes_to_cluster:
            The nodes that we want to cluster into subtrees
        function:
            The function according to which "flatness" is measured
        Returns
        -------
        subtrees:
            The subtrees created
        statistics:
            A dictionary of statistical measures used to construct the subtrees
        """
        # Will contain details of the
        # statistics used to perform clustering
        statistics = dict()

        # Find the critical paths between the nodes to cluster on the tree
        critical_paths = FunctionTopology.critical_paths(tree, nodes_to_cluster)

        # Get the critical path curves (the values of the function
        # along each curve, relative to the maximum endpoint)
        curves = dict()
        min_function = min(function)
        for key in critical_paths:
            path = critical_paths[key]
            curve = function[path] - min_function
            curve /= max(curve[0], curve[-1])
            curves[key] = curve
        statistics["curves"] = curves

        def deviation(curve):
            return max(abs(c - 1.0) for c in curve)

        # Get the "deviation" of each curve
        deviations = {key: deviation(curves[key]) for key in curves}
        statistics["deviations"] = deviations

        # Values to create a KDE for the distribution of
        kde_vals = [deviations[key] for key in deviations]
        kde_vals.sort()

        kde_bandwidth = 0.01
        if len(kde_vals) > 1:
            # Optimal value for gaussian distributions with gaussian kernel (perhaps non-optimal here) from
            # Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis.
            # London: Chapman & Hall/CRC. p. 45. ISBN 978-0-412-24620-3.
            kde_bandwidth = 1.06 * np.std(kde_vals) * (len(kde_vals) ** (-0.2))

            # Rule-of-thumb value from wikipedia
            # iqr = np.percentile(kde_vals, [75, 25])
            # iqr = iqr[0] - iqr[1]
            # kde_bandwidth = 0.9 * min(np.std(kde_vals), iqr / 1.34) * (len(kde_vals) ** (-0.2))

        statistics["kde_bandwidth"] = kde_bandwidth

        # Create a KDE for the distribution of deviations
        kde_points = 4000
        kde_xs = np.linspace(0, 1, kde_points)
        kde_ys = np.zeros(kde_points)
        for v in kde_vals:
            kde_ys += np.exp(-((kde_xs - v) / kde_bandwidth) ** 2)
        statistics["kde_xs"] = kde_xs
        statistics["kde_ys"] = kde_ys

        # Evaluate a threshold for deviation of curves,
        # below which curves are considered to be "flat"
        deviation_threshold = 0.0
        kde_threshold = max(kde_ys) / 10.0
        for i in range(kde_points):
            if kde_ys[i] < kde_threshold:
                while i > 0 and kde_ys[i - 1] < kde_ys[i]:
                    i -= 1
                while i < kde_points and kde_ys[i + 1] < kde_ys[i]:
                    i += 1
                deviation_threshold = kde_xs[i]
                break
        statistics["deviation_threshold"] = deviation_threshold

        # Use network cleaving value, if provided
        if self.settings.network_cleaving_threshold is not None:
            deviation_threshold = self.settings.network_cleaving_threshold

        # Build a graph from the "flat" critical paths
        edges = []
        for key in critical_paths:
            if deviations[key] <= deviation_threshold:
                path = critical_paths[key]
                if any(i in self.convex_hull for i in path):
                    continue
                edges.extend((path[i - 1], path[i]) for i in range(1, len(path)))
        graph = networkx.Graph()
        graph.add_edges_from(edges)
        subgraphs = tuple(networkx.Graph(networkx.subgraph(graph, c)) for c in connected_components(graph))

        return subgraphs, statistics

    #############
    # UTILITIES #
    #############

    def print(self, to_print) -> None:
        """
        Print the given information, if self.print_debug is True.
        """
        if self.print_method is not None:
            self.print_method(to_print)

    def general_assertions(self):
        """
        Assertions that should be true for any topology.
        """
        with TimeAndStore(self._stored_properties, "Performing general assertions") as stored:
            if stored.value is not None:
                return stored.value

            # All points should be in these graphs
            for i in range(len(self.points)):
                assert i in self.graph
                assert i in self.climb_graph
                assert i in self.max_spanning_tree
                assert i in self.min_deviation_tree

            # The critical trees should contain all local maxima in the bulk
            for i in self.bulk_maxima:
                assert i in self.critical_tree
                assert i in self.stationary_tree

            # Local maxima should actually be local maxima
            for i in self.local_maxima:
                for j in self.graph[i]:
                    if self.function[j] > self.function[i]:
                        raise Exception(f"Neighbour of local maximum is greater: "
                                        f"{self.function[j]} > {self.function[i]}")

            # Each maxima family should be connected
            for family in self.extended_maxima_families:
                subgraph = networkx.subgraph(self.graph, self.extended_maxima_families[family])
                assert len(list(connected_components(subgraph))) == 1

            # Regions should correspond to maxima families
            # or to the convex hull (0)
            for i in self.regions:
                assert i in self.extended_maxima_families or i == 0

            stored.value = True
            return stored.value

    #########
    # PLOTS #
    #########

    def show_plots(self, plots=None, png_name: str = None) -> None:
        """
        Generate the given suite of plots (or that specified in the settings).
        """
        plots = plots or self.settings.plots

        if len(plots) == 0:
            return  # Nothing to plot

        with TimeAndStore(self._stored_properties, "Generating plots"):

            import matplotlib.pyplot as plt
            plt.switch_backend("TkAgg")
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] + get_color_alphabet()

            def create_axes(title, axis_limit=10, show_axes=True):

                fig = plt.figure()
                fig.suptitle(title)
                fig.canvas.manager.set_window_title(title)

                if self.dimensions == 3:
                    ax = fig.add_subplot(111, projection="3d")
                    ax.set_xlim((-axis_limit, axis_limit))
                    ax.set_ylim((-axis_limit, axis_limit))
                    ax.set_zlim((-axis_limit, axis_limit))
                    if not show_axes:
                        ax._axis3don = False
                    return ax

                if self.dimensions == 2:
                    ax = fig.add_subplot(111)
                    if show_axes:
                        ax.set_xlim((min(self.points[:, 0]), max(self.points[:, 0])))
                        ax.set_ylim((min(self.points[:, 1]), max(self.points[:, 1])))
                    else:
                        ax.set_axis_off()
                    return ax

                raise Exception(f"Can't plot in {self.dimensions} dimensions!")

            def scatter_points(ax, indices, max_points=1000,
                               colormap: Optional[dict] = None,
                               discrete_colormap: Optional[dict] = None,
                               **kwargs):

                if len(indices) == 0:
                    return  # Nothing to scatter

                # Probability of keeping a particular point
                prob = max_points / len(indices)
                indices = [i for i in indices if np.random.random() < prob]
                c = None if colormap is None else [colormap[i] for i in indices]

                if discrete_colormap is not None:
                    c = dict()
                    for i, v in enumerate(sorted({discrete_colormap[k] for k in discrete_colormap})):
                        c[v] = color_cycle[i % len(color_cycle)]
                    c = [c[discrete_colormap[i]] for i in indices]

                if self.dimensions == 3:
                    ax.scatter(self.points[indices, 0], self.points[indices, 1], self.points[indices, 2], c=c, **kwargs)
                elif self.dimensions == 2:
                    ax.scatter(self.points[indices, 0], self.points[indices, 1], c=c, **kwargs)
                else:
                    raise Exception(f"Can't plot in {self.dimensions} dimensions!")

            def plot_edge(ax, edge, color=None, alpha=1.0):
                color = color or "black"
                if self.dimensions == 3:
                    ax.plot3D(self.points[edge, 0], self.points[edge, 1], self.points[edge, 2],
                              color=color, alpha=alpha)
                elif self.dimensions == 2:
                    ax.plot(self.points[edge, 0], self.points[edge, 1],
                            color=color, alpha=alpha)
                else:
                    raise Exception(f"Can't plot in {self.dimensions} dimensions!")

            def plot_graph(ax, graph: networkx.Graph, color=None, scatter_maxima=True, alpha=1.0):
                for edge in graph.edges:
                    plot_edge(ax, edge, color=color, alpha=alpha)
                if scatter_maxima:
                    scatter_points(ax, self.bulk_maxima, color="black")

            def plot_subgraphs(ax, graphs):

                for i, graph in enumerate(graphs):
                    col = color_cycle[i % len(color_cycle)]
                    maxima = [lm for lm in graph.nodes if lm in self.bulk_maxima]
                    plot_graph(ax, graph, color=col, scatter_maxima=False)
                    scatter_points(ax, maxima, colormap={i: col for i in maxima})

                def is_isolated(lm):
                    for graph in graphs:
                        if lm in graph:
                            return False
                    return True

                scatter_points(ax, [lm for lm in self.bulk_maxima if is_isolated(lm)], color="black")

            if TopologyPlot.X_AXIS_VALUES in plots:
                def gen_point(x):
                    pt = [0] * self.dimensions
                    pt[0] = x
                    return pt

                xs = np.linspace(min(self.points[:, 0]), max(self.points[:, 0]), 1000)
                ys = [self.function[self.nearest_point(gen_point(x))] for x in xs]
                fig = plt.figure()
                fig.suptitle("Function values along x-axis")
                ax = fig.add_subplot(111)
                ax.plot(xs, ys)

            if TopologyPlot.ALL_POINTS in plots:
                all_pts = range(len(self.points))
                ax = create_axes("Points (colored by region)", show_axes=self.dimensions != 2)
                scatter_points(ax, all_pts, discrete_colormap={i: self.regions[i] for i in all_pts}, max_points=1000)

                if self.dimensions == 2:
                    ax.set_aspect(1.0)
                    max_grad = max(np.linalg.norm(g) for g in self.gradient)
                    max_length = np.mean(
                        [np.linalg.norm(self.points[j] - self.points[i])
                         for i in range(len(self.points)) for j in self.graph[i]])

                    for i in range(len(self.points)):
                        p = self.points[i]
                        g = self.gradient[i]
                        g = max_length * g / max_grad
                        ax.plot([p[0], p[0] + g[0]], [p[1], p[1] + g[1]], color="black")

                    def grad_dot(a, b):
                        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

                    if self._analytic_gradients is not None:
                        fig = plt.figure()
                        fig.suptitle("Numerical gradient quality")
                        ax = fig.add_subplot(111)
                        ax.set_yscale("log")
                        ax.hist([grad_dot(a, b) for a, b in zip(self.gradient, self.numerical_gradients)],
                                bins=100, color="grey")
                        ax.set_ylabel("Count")
                        ax.set_xlabel(r"$\frac{g\cdot \nabla f}{|g||\nabla f|}$")

            if TopologyPlot.BULK_AND_EXTENDED_MAXIMA in plots:
                ax = create_axes("Bulk/Extended maxima")
                scatter_points(ax, self.bulk_maxima, color="black", label="Local maxima")
                for family in self.extended_maxima_families:
                    scatter_points(ax, self.extended_maxima_families[family], alpha=0.3)
                ax.view_init(elev=10.0, azim=0.0)
                ax.dist = 2.0

            if TopologyPlot.EXTENDED_MAXIMA_FAMILIES in plots:
                ax = create_axes("Extended maxima families")
                for family in self.extended_maxima_families:
                    scatter_points(ax, self.extended_maxima_families[family], alpha=0.3)

            if TopologyPlot.CRITICAL_TREE in plots:
                plot_graph(create_axes("Critical tree"), self.critical_tree)
            if TopologyPlot.CLEAVED_CRITICAL_TREE in plots:
                plot_subgraphs(create_axes("Cleaved critical tree"), self.cleaved_critical_tree[0])
            if TopologyPlot.CRITICAL_NETWORK in plots:
                plot_graph(create_axes("Critical network"), self.critical_network)

            if TopologyPlot.STATIONARY_TREE in plots:
                plot_graph(create_axes("Stationary tree"), self.stationary_tree)
            if TopologyPlot.CLEAVED_STATIONARY_TREE in plots:
                plot_subgraphs(create_axes("Cleaved stationary tree"), self.cleaved_stationary_tree[0])

            if TopologyPlot.CONVEX_HULL in plots:
                scatter_points(create_axes("Convex hull"), self.convex_hull)

            if TopologyPlot.SADDLE_LINKED_REGION_GRAPH in plots:
                plot_graph(create_axes("Saddle-linked regions"), self.saddle_linked_region_graph)

            if TopologyPlot.REGION_PATHS in plots:
                ax = create_axes("Region paths")
                for region in self.region_centres:
                    scatter_points(ax, [self.region_centres[region]])
                for index, (i, j) in enumerate(self.region_paths):
                    path = self.region_paths[i, j]
                    color = color_cycle[index % len(color_cycle)]
                    for n in range(1, len(path)):
                        plot_edge(ax, [path[n - 1], path[n]], color=color)

            if TopologyPlot.LOCAL_MAXIMA_VALUES_HIST in plots:
                fig = plt.figure()
                title = "Local maxima values"
                fig.suptitle(title)
                fig.canvas.manager.set_window_title(title)
                ax = fig.add_subplot(211)
                ax.set_title("Local maxima")
                ax.hist([self.function[n] for n in self.local_maxima], bins=100)
                ax = fig.add_subplot(212)
                ax.set_title("Bulk maxima")
                ax.hist([self.function[n] for n in self.bulk_maxima], bins=100)

            if TopologyPlot.STATIONARY_PATH_STATS in plots:
                self.plot_stationary_path_stats(plt)

        if png_name is not None:
            plt.savefig(png_name + ".png")
        else:
            plt.show()

    def show_all_plots(self):
        self.show_plots(TopologyPlot.all())

    def plot_stationary_path_stats(self, plt, title=None):

        def plot_path_curves(ax, title, statistics):
            # Plot path curves
            ax.set_title(title)
            curves = statistics["curves"]
            deviations = statistics["deviations"]
            deviation_threshold = statistics["deviation_threshold"]
            for key in curves:
                ys = curves[key]
                ys = ys if ys[-1] < ys[0] else list(reversed(ys))
                xs = [i / (len(ys) - 1.0) for i in range(len(ys))]
                dev = deviations[key]
                linestyle = "solid" if dev < deviation_threshold else "dotted"
                dev = min(1.0, dev)
                ax.plot(xs, ys, color=(dev, 0, 1 - dev), linestyle=linestyle)

        def plot_deviation_histogram(ax, title, statistics):

            if len(statistics["deviations"]) == 0:
                return

            vals = statistics["deviations"]
            vals = [vals[k] for k in vals]
            deviation_threshold = statistics["deviation_threshold"]
            kde_x, kde_y = statistics["kde_xs"], statistics["kde_ys"]

            # Plot histogram of deviations of path curves
            ax.set_title(title)
            y, x, _ = ax.hist(vals, bins=1000, range=[0.0, max(vals)])
            ax.axvline(deviation_threshold, color="red", label=f"Threshold = {deviation_threshold:.3}")
            kde_y *= max(y) / max(kde_y)

            kde_maxima = [i for i in range(1, len(kde_y) - 1) if kde_y[i - 1] < kde_y[i] and kde_y[i + 1] < kde_y[i]]
            for i in kde_maxima:
                ax.axvline(kde_x[i], color="green")

            kde_bandwitdh = statistics["kde_bandwidth"]
            ax.plot(kde_x, kde_y, color="blue", label=f"KDE (bandwidth = {kde_bandwitdh:.3})")
            ax.set_xlim((0, 0.01 + max(vals)))
            ax.legend()

        fig = plt.figure()
        title = title or "Stationary path statistics"
        fig.suptitle(title)
        fig.canvas.set_window_title(title)

        stats_stationary = self.cleaved_stationary_tree[1]
        stats_maxima = self.cleaved_critical_tree[1]
        plot_path_curves(fig.add_subplot(221), "Stationary path curves", stats_stationary)
        plot_path_curves(fig.add_subplot(222), "Maxima path curves", stats_maxima)
        plot_deviation_histogram(fig.add_subplot(223), "Deviation of stationary path curves", stats_stationary)
        plot_deviation_histogram(fig.add_subplot(224), "Deviation of maxima path curves", stats_maxima)


class TimeAndStore:
    print_method = None

    def __init__(self, dictionary: dict, description):
        self.description = description
        self.dictionary = dictionary
        self._value = None

    @property
    def value(self):
        if self._value is not None:
            return self._value
        if self.description not in self.dictionary:
            return None
        return self.dictionary[self.description]

    @value.setter
    def value(self, val):
        self._value = val
        self.dictionary[self.description] = val

    def __enter__(self):
        self.start_time = time()
        self.first_time = self.value is None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time() - self.start_time
        if self.first_time and TimeAndStore.print_method is not None:
            TimeAndStore.print_method(self.description + f" took {elapsed:.3f} seconds.")


def get_color_alphabet() -> List[Tuple[float, float, float]]:
    """
    The color alphabet (excluding white) from P. Green-Armytage (2010).
    "A Colour Alphabet and the Limits of Colour Coding".
    Colour: Design & Creativity. 5 (10): 123
    """
    return [
        (0.941, 0.639, 1.000),
        (0.000, 0.459, 0.863),
        (0.600, 0.247, 0.000),
        (0.298, 0.000, 0.361),
        (0.098, 0.098, 0.098),
        (0.000, 0.361, 0.192),
        (0.169, 0.808, 0.282),
        (1.000, 0.800, 0.600),
        (0.502, 0.502, 0.502),
        (0.580, 1.000, 0.710),
        (0.561, 0.486, 0.000),
        (0.616, 0.800, 0.000),
        (0.761, 0.000, 0.533),
        (0.000, 0.200, 0.502),
        (1.000, 0.643, 0.020),
        (1.000, 0.659, 0.733),
        (0.259, 0.400, 0.000),
        (1.000, 0.000, 0.063),
        (0.369, 0.945, 0.949),
        (0.000, 0.600, 0.561),
        (0.878, 1.000, 0.400),
        (0.455, 0.039, 1.000),
        (0.600, 0.000, 0.000),
        (1.000, 1.000, 0.502),
        (1.000, 1.000, 0.000),
        (1.000, 0.314, 0.020)
    ]


def plot_saved_topology(filename=None):
    if filename is None:
        for f in os.listdir("."):
            if f.endswith("topology.save"):
                filename = f
                break

    t = FunctionTopology.load_from_disk(filename)
    t.show_plots()
