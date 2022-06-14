import os
import numpy as np
from topgrid.FunctionTopology import FunctionTopology
from topgrid.TopologyInput import TopologyInput, GraphSteepestAscentMethod
import pytest


def create_topology(points, function, **kwargs):
    t = FunctionTopology(points, function, **kwargs)
    t.general_assertions()
    return t


def gaussian_topology(seed=1024, count=1000, **kwargs):
    np.random.seed(seed)
    points = (np.random.random((count, 3)) - 0.5) * 2
    function = np.exp(-np.linalg.norm(points, axis=1) ** 2)
    return create_topology(points, function, **kwargs)


def double_gaussian_topology(seed=1024, count=1000, **kwargs):
    np.random.seed(seed)
    points = (np.random.random((count, 3)) - 0.5) * 4
    dp1 = points - np.array((1.0, 0.0, 0.0))
    dp2 = points + np.array((1.0, 0.0, 0.0))
    function = \
        np.exp(-np.linalg.norm(dp1, axis=1) ** 2) + \
        np.exp(-np.linalg.norm(dp2, axis=1) ** 2)
    return create_topology(points, function, **kwargs)


def double_gaussian_topology_2d(edge_count=21,
                                a=(0.5, 0.25), b=(-0.5, -0.25),
                                analytic_gradient=False, random_grid=False, **kwargs):
    xs = np.linspace(-1, 1, edge_count)
    points = np.array([(x, y) for x in xs for y in xs])

    if random_grid:
        np.random.seed(1024)
        points = np.random.random(points.shape) * 2 - 1

    assert points.shape == (edge_count * edge_count, 2)

    s = 0.5

    f1 = np.exp(-np.linalg.norm(points - a, axis=1) ** 2 / s ** 2)
    f2 = np.exp(-np.linalg.norm(points - b, axis=1) ** 2 / s ** 2)

    g1 = -2 * (points - a) * f1[:, np.newaxis] / s ** 2
    g2 = -2 * (points - b) * f2[:, np.newaxis] / s ** 2

    if analytic_gradient:
        return create_topology(points, f1 + f2, gradients=g1 + g2, **kwargs)
    return create_topology(points, f1 + f2, **kwargs)


def fake_atomic_system(centres, atomic_numbers, **kwargs):
    points_normal_dist = kwargs.pop("points_normal_dist", True)
    points_per_atom = kwargs.pop("points_per_atom", 500)
    remap_negative = kwargs.pop("remap_negative", False)

    centres = [np.array(c) for c in centres]

    if points_normal_dist:
        points = [np.random.normal(loc=c, size=(points_per_atom, 3)) for c in centres]
        points = np.concatenate(points)
    else:
        points = (np.random.random((points_per_atom * len(centres), 3)) * 2 - 1) * 3

    function = np.zeros(len(points))
    for c, a in zip(centres, atomic_numbers):
        dp = points - c
        function += a * np.exp(-16 * np.linalg.norm(dp, axis=1) ** 2)

    # Remap the function so that it's maximum is at 0
    if remap_negative:
        function -= max(function)

    return create_topology(points, function, **kwargs)


def fake_benzene(seed=1024, **kwargs):
    np.random.seed(seed)

    # Set carbon positions
    x = np.cos(np.pi / 6.0)
    y = np.sin(np.pi / 6.0)
    centres = [
        [0, 1, 0],
        [0, -1, 0],
        [x, y, 0],
        [x, -y, 0],
        [-x, y, 0],
        [-x, -y, 0],
    ]

    # Add hydrogen positions
    for i in range(6):
        centres.append([x * 2 for x in centres[i]])

    atomic_numbers = [6] * 6 + [1] * 6
    return fake_atomic_system(centres, atomic_numbers, **kwargs)


def test_gaussian():
    t = gaussian_topology()
    assert t.region_count == 1
    return t


def test_double_gaussian():
    t = double_gaussian_topology()
    assert len(t.bulk_maxima) > 0
    assert t.region_count == 2
    return t


def test_function_update():
    t1 = gaussian_topology()
    t2 = double_gaussian_topology()
    assert t1.region_count == 1
    assert t2.region_count == 2
    t1.function = t2.function
    assert t1.region_count == 2
    return t1


def test_benzene():
    t = fake_benzene()
    assert t.region_count == 12
    assert len(t.region_paths) == 12
    return t


def test_benzene_negative():
    t = fake_benzene(remap_negative=True)
    assert t.region_count == 12
    return t


def test_flat():
    np.random.seed(1024)
    points = np.random.random((500, 3)) * 2 - 1
    function = np.zeros(500)
    points[0] = (0, 0, 0)
    function[0] = 1.0
    t = create_topology(points, function)
    assert t.region_count == 1
    return t


def test_extended_maxima():
    np.random.seed(1024)
    coords = np.random.random((8000, 3)) * 2 - 1
    function = np.exp(-4 * (np.linalg.norm(coords, axis=1) - 0.5) ** 2)
    t = create_topology(coords, function)
    assert t.region_count == 1
    return t


def test_two_extended_maxima():
    np.random.seed(1024)
    coords = np.random.random((8000, 3)) * 4 - 2
    r = np.linalg.norm(coords, axis=1)
    function = np.exp(-4 * (r - 2) ** 2) + 0.5 * np.exp(-4 * (r - 1) ** 2)
    t = create_topology(coords, function)
    assert t.region_count == 2
    return t


def test_two_extended_maxima_shallow():
    np.random.seed(1024)
    coords = np.random.random((8000, 3)) * 4 - 2
    r = np.linalg.norm(coords, axis=1)
    function = np.exp(-4 * (r - 1.8) ** 2) + np.exp(-4 * (r - 1) ** 2)
    t = create_topology(coords, function)
    return t


def test_two_extended_maxima_same_value():
    np.random.seed(1024)
    coords = np.random.random((8000, 3)) * 4 - 2
    function = np.exp(-4 * (np.linalg.norm(coords, axis=1) - 2) ** 2)
    function += np.exp(-4 * (np.linalg.norm(coords, axis=1) - 1) ** 2)
    t = create_topology(coords, function)
    assert t.region_count == 2
    return t


def test_three_extended_maxima():
    np.random.seed(1024)
    coords = np.random.random((8000, 3)) * 8 - 4
    r = np.linalg.norm(coords, axis=1)
    function = 0.25 * np.exp(-4 * (r - 3) ** 2) + \
               0.50 * np.exp(-4 * (r - 2) ** 2) + \
               1.00 * np.exp(-4 * (r - 1) ** 2)
    t = create_topology(coords, function)
    assert t.region_count == 3
    return t


def test_maxima_on_hull():
    np.random.seed(1024)
    coords = np.random.random((8000, 3)) * 4 - 2
    r = np.linalg.norm(coords, axis=1)
    function = np.exp(-r ** 2) - np.exp(-4 * r ** 2)
    function = 1 - function
    t = create_topology(coords, function)
    assert t.region_count == 2
    return t


def test_broken_symmetry():
    # Generate a spherically-symmetric grid
    points = []
    function = []
    for r in np.linspace(0.01, 1, 16):
        for theta in np.linspace(0.01, np.pi - 0.01, 16):
            for phi in np.linspace(0.01, 2 * np.pi - 0.01, 16):
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                points.append((x, y, z))

                # Generate a maxima shell that has
                # been squished in the z direction
                r_eff = r + 2 * np.sqrt(z * z)
                function.append(np.exp(-(r_eff - 0.5) ** 2))

    t = create_topology(points, function)
    assert t.region_count == 1
    return t


def test_maxima_lines():
    np.random.seed(1024)
    coords = np.random.random((8000, 3)) * 2 - 1
    xy_radii = np.linalg.norm(coords[:, :2], axis=1)
    sq_dist_from_circle = (xy_radii - 0.5) ** 2 + coords[:, 2] ** 2
    function = np.maximum(np.exp(-sq_dist_from_circle), np.exp(-xy_radii ** 2))
    inp = TopologyInput()
    inp.stationary_value_tolerance = 0.01
    t = create_topology(coords, function, settings=inp)
    assert t.region_count == 2
    return t


def test_shallow_isolated_maxima():
    np.random.seed(1024)
    coords = np.random.random((8000, 3)) * 2 - 1
    c1 = np.array((0.5, 0, 0))
    c2 = np.array((-0.5, 0, 0))
    width = 0.6
    function = np.exp(-np.linalg.norm(coords - c1, axis=1) ** 2 / width ** 2) + \
               np.exp(-np.linalg.norm(coords - c2, axis=1) ** 2 / width ** 2)
    t = create_topology(coords, function)
    t.general_assertions()
    return t


def test_stored_property():
    t = double_gaussian_topology()
    assert t.stored_properties_count > 0
    t.clear_stored_properties()
    assert t.stored_properties_count == 0


def test_region_charges():
    t = double_gaussian_topology()
    charges = t.region_charges([1.0] * len(t.points))
    assert len(charges) == 2
    assert abs(charges[1] - 81.8156) < 10e-3
    assert abs(charges[2] - 78.5469) < 10e-3


def test_1d():
    points = [[0.0], [1.0], [2.0]]
    function = [0.0, 1.0, 2.0]
    t = create_topology(points, function)
    assert t.convex_hull == {0, 2}
    assert set(t.graph[0]) == {1}
    assert set(t.graph[1]) == {0, 2}
    assert set(t.graph[2]) == {1}


def test_gradient_1d():
    points = [[0.0], [1.0], [2.0]]
    function = [0.0, 1.0, 0.0]
    t = create_topology(points, function)
    assert np.linalg.norm(t.gradient[0] - [1.0]) < 10e-6
    assert np.linalg.norm(t.gradient[1] - [0.0]) < 10e-6


def test_on_graph_1d():
    points = np.linspace(0, 1, 21)
    function = abs(np.sin(points * np.pi * 2))
    points = [[x] for x in points]
    t = create_topology(points, function)
    assert t.region_count == 2


def test_off_graph_1d():
    settings = TopologyInput()
    settings.graph_ascent_method = GraphSteepestAscentMethod.OFF_GRAPH
    points = np.linspace(0, 1, 21)
    function = abs(np.sin(points * np.pi * 2))
    points = [[x] for x in points]
    t = create_topology(points, function, settings=settings)
    assert t.region_count == 2


def test_on_graph_2d():
    t = double_gaussian_topology_2d()
    assert t.region_count == 2


def test_off_graph_2d_analytic():
    settings = TopologyInput()
    settings.graph_ascent_method = GraphSteepestAscentMethod.OFF_GRAPH
    t = double_gaussian_topology_2d(settings=settings, analytic_gradient=True)
    assert t.region_count == 2


def test_off_graph_2d_numerical():
    settings = TopologyInput()
    settings.graph_ascent_method = GraphSteepestAscentMethod.OFF_GRAPH
    t = double_gaussian_topology_2d(settings=settings)
    assert t.region_count == 2


def test_off_graph_2d_analytic_random_grid():
    settings = TopologyInput()
    settings.graph_ascent_method = GraphSteepestAscentMethod.OFF_GRAPH
    t = double_gaussian_topology_2d(settings=settings, analytic_gradient=True, random_grid=True)
    assert t.region_count == 2


def test_off_graph_2d_numerical_random_grid():
    settings = TopologyInput()
    settings.graph_ascent_method = GraphSteepestAscentMethod.OFF_GRAPH
    t = double_gaussian_topology_2d(settings=settings, random_grid=True)
    assert t.region_count == 2


def test_off_graph():
    settings = TopologyInput()
    settings.graph_ascent_method = GraphSteepestAscentMethod.OFF_GRAPH
    t = double_gaussian_topology(settings=settings)
    assert t.region_count == 2


path_stats_cases = [
    test_two_extended_maxima_shallow,
    test_benzene,
    test_extended_maxima,
    test_three_extended_maxima,
    test_shallow_isolated_maxima,
    test_maxima_lines
]


def run_path_stat_case(i):
    import matplotlib.pyplot as plt
    plt.switch_backend("TkAgg")
    t = path_stats_cases[i]()
    t.plot_stationary_path_stats(plt, path_stats_cases[i].__name__)
    plt.show()


@pytest.mark.skip(reason="This is intended to be run interactively")
def test_path_statistics():
    from multiprocessing import Pool
    Pool(len(path_stats_cases)).map(run_path_stat_case, range(len(path_stats_cases)))
