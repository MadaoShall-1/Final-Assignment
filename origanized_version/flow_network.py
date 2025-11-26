"""
Flow Network Implementation
Provides the FlowNetwork class for maximum flow computations.
"""

from collections import defaultdict, deque
from typing import Dict, Optional, Set


class FlowNetwork:
    """
    Represents a flow network for the maximum flow problem.
    Uses adjacency list representation with residual capacities.
    """
    
    def __init__(self):
        """Initialize an empty flow network."""
        self.graph = defaultdict(lambda: defaultdict(int))
        self.flow = defaultdict(lambda: defaultdict(int))
        
    def add_edge(self, u: str, v: str, capacity: int):
        """
        Add an edge to the flow network.
        
        Args:
            u: Source vertex
            v: Destination vertex
            capacity: Edge capacity
        """
        self.graph[u][v] += capacity
        
    def get_residual_capacity(self, u: str, v: str) -> int:
        """
        Calculate residual capacity of edge (u, v).
        
        Args:
            u: Source vertex
            v: Destination vertex
            
        Returns:
            Residual capacity of the edge
        """
        return self.graph[u][v] - self.flow[u][v]
    
    def find_augmenting_path_bfs(self, source: str, sink: str) -> Optional[Dict[str, str]]:
        """
        Find an augmenting path from source to sink using BFS.
        
        Args:
            source: Source vertex
            sink: Sink vertex
            
        Returns:
            Parent dictionary representing the path, or None if no path exists
        """
        visited = {source}
        queue = deque([source])
        parent = {}
        
        while queue:
            u = queue.popleft()
            
            for v in self.graph[u]:
                if v not in visited and self.get_residual_capacity(u, v) > 0:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
                    
                    if v == sink:
                        return parent
                        
        return None
    
    def ford_fulkerson(self, source: str, sink: str) -> int:
        """
        Implement Ford-Fulkerson algorithm to find maximum flow.
        
        Args:
            source: Source vertex
            sink: Sink vertex
            
        Returns:
            Maximum flow value
        """
        max_flow = 0
        
        while True:
            parent = self.find_augmenting_path_bfs(source, sink)
            
            if parent is None:
                break
                
            # Find minimum capacity along the path
            path_flow = float('inf')
            v = sink
            
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, self.get_residual_capacity(u, v))
                v = u
                
            # Update flow along the path
            v = sink
            while v != source:
                u = parent[v]
                self.flow[u][v] += path_flow
                self.flow[v][u] -= path_flow
                v = u
                
            max_flow += path_flow
            
        return max_flow
    
    def get_min_cut(self, source: str) -> Set[str]:
        """
        Find vertices reachable from source in residual graph.
        
        Args:
            source: Source vertex
            
        Returns:
            Set of reachable vertices (source side of min cut)
        """
        visited = set()
        queue = deque([source])
        visited.add(source)
        
        while queue:
            u = queue.popleft()
            
            for v in self.graph[u]:
                if v not in visited and self.get_residual_capacity(u, v) > 0:
                    visited.add(v)
                    queue.append(v)
                    
        return visited
