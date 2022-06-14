# TOPGRID

Standalone version of the topology analysis in the [QUEST](https://quest.codes/) quantum chemistry code.

Algorithms for analysing topographical and topological quantities 
of a function defined on an arbitrary set of points in space.

## Usage
A good place to start is to make sure the tests work on your system - see the testing section below.

The `FunctionTopology` class, defined in `FunctionTopology.py` should 
be given a set of points and function evaluations at those points, like so:

```python
from topgrid.FunctionTopology import FunctionTopology
import numpy as np

# Build 1000 random points in 3D space with coordinates in [-1,1]
points = (np.random.random((1000, 3)) - 0.5) * 2

# Evaluate a guassian function on the above points
function = np.exp(-np.linalg.norm(points, axis=1) ** 2)

# Construct a FunctionTopology object for the above points/function
top = FunctionTopology(points, function)
```
Topological/topographical quantities of the function can then be accessed 
as properties of the `FunctionTopology` object, like so:

```python
# Print the number of basins of attraction of the function
print(top.region_count)
```
Some pre-defined plots can also be produced, either all together as one 
suite of plots, or by selecting specific plots defined in the `TopologyPlot` enum:
```python
from topgrid.TopologyPlot import TopologyPlot

# Show the suite of all pre-defined plots
top.show_all_plots()

# Show a subset of the pre-defined plots
top.show_plots([TopologyPlot.EXTENDED_MAXIMA_FAMILIES, TopologyPlot.CRITICAL_TREE])
```

## Testing
To check if the algorithms work on your system, the test suite should be used. 
To run the test suite, simply type `pytest` in the root directory:
```commandline
mick@micks-pc:~/programming/topgrid$ pytest
========================= test session starts ==========================
platform linux -- Python 3.8.10, pytest-6.2.5, py-1.11.0, pluggy-1.0.0
rootdir: /home/mick/programming/topgrid
plugins: xdist-2.4.0, forked-1.3.0
collected 28 items                                                     

test/test_FunctionTopology.py ...........................s       [100%]

==================== 27 passed, 1 skipped in 24.33s ====================
```