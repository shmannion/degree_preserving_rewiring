# Installation

## From source

```console
git clone https://github.com/Shaneul/degree_preserving_rewiring
cd degree_preserving_rewiring
pip install .

```

# Example

Simple example rewiring a random graph.
```python
import networkx as nx
from degree_preserving_rewiring import rewire

G = nx.erdos_renyi_graph(5000, p = 5/4999, seed = 42)
G, results = rewire(G, 0.4, 'random graph', sample_size=20, timed=True, 
                    time_limit=120, method='new')

```

# Info
For details, see paper on arXiv 
