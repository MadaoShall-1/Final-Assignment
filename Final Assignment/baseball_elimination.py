"""
Baseball Elimination Algorithm Implementation
CS 5800 Spring 2025 Final Project

This module implements the baseball elimination algorithm using maximum flow
to determine if a team has been mathematically eliminated from winning their division.

Authors: [Your Group Names Here]
"""

import sys
import time
import random
import numpy as np
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Set

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


class BaseballElimination:
    """
    Solves the baseball elimination problem using network flow.
    """
    
    def __init__(self, teams: List[str], wins: List[int], losses: List[int], 
                 remaining: List[int], games: List[List[int]]):
        """
        Initialize with league standings.
        
        Args:
            teams: List of team names
            wins: List of wins for each team
            losses: List of losses for each team
            remaining: List of remaining games for each team
            games: Matrix where games[i][j] = remaining games between teams i and j
        """
        self.teams = teams
        self.wins = wins
        self.losses = losses
        self.remaining = remaining
        self.games = games
        self.n = len(teams)
        
        # Validate input
        self._validate_input()
        
    def _validate_input(self):
        """Validate that input data is consistent."""
        assert len(self.teams) == len(self.wins) == len(self.losses) == len(self.remaining)
        assert len(self.games) == self.n
        
        for i in range(self.n):
            assert len(self.games[i]) == self.n
            assert self.games[i][i] == 0  # No team plays itself
            
        # Check symmetry of games matrix
        for i in range(self.n):
            for j in range(i + 1, self.n):
                assert self.games[i][j] == self.games[j][i], \
                    f"Games matrix not symmetric: games[{i}][{j}]={self.games[i][j]} != games[{j}][{i}]={self.games[j][i]}"
        
        # IMPORTANT: Verify that remaining games matches the games matrix
        for i in range(self.n):
            total_games_for_team_i = sum(self.games[i])
            assert total_games_for_team_i == self.remaining[i], \
                f"Team {self.teams[i]}: remaining={self.remaining[i]} doesn't match sum of games in matrix={total_games_for_team_i}"
                
    def is_trivially_eliminated(self, team_idx: int) -> bool:
        """
        Check if a team is trivially eliminated.
        
        A team is trivially eliminated if even winning all remaining games
        leaves them with fewer wins than some other team already has.
        
        Args:
            team_idx: Index of team to check
            
        Returns:
            True if trivially eliminated
        """
        max_possible_wins = self.wins[team_idx] + self.remaining[team_idx]
        
        for i in range(self.n):
            if i != team_idx and self.wins[i] > max_possible_wins:
                return True
                
        return False
    
    def build_flow_network(self, team_idx: int) -> Tuple[FlowNetwork, int]:
        """
        Build flow network to check if team is eliminated.
        
        Args:
            team_idx: Index of team to check
            
        Returns:
            Tuple of (flow network, total games to distribute)
        """
        network = FlowNetwork()
        max_wins = self.wins[team_idx] + self.remaining[team_idx]
        
        # Create vertex names
        source = "source"
        sink = "sink"
        
        # Count total games between other teams
        total_games = 0
        
        # Add edges from source to game vertices
        for i in range(self.n):
            if i == team_idx:
                continue
                
            for j in range(i + 1, self.n):
                if j == team_idx:
                    continue
                    
                if self.games[i][j] > 0:
                    game_vertex = f"game_{i}_{j}"
                    # Edge from source to game vertex
                    network.add_edge(source, game_vertex, self.games[i][j])
                    total_games += self.games[i][j]
                    
                    # Edges from game vertex to team vertices
                    team_i_vertex = f"team_{i}"
                    team_j_vertex = f"team_{j}"
                    network.add_edge(game_vertex, team_i_vertex, float('inf'))
                    network.add_edge(game_vertex, team_j_vertex, float('inf'))
        
        # Add edges from team vertices to sink
        for i in range(self.n):
            if i == team_idx:
                continue
                
            team_vertex = f"team_{i}"
            capacity = max(0, max_wins - self.wins[i])
            network.add_edge(team_vertex, sink, capacity)
            
        return network, total_games
    
    def is_eliminated(self, team_idx: int) -> Tuple[bool, Optional[List[str]]]:
        """
        Check if a team is eliminated and find certificate if so.
        
        Args:
            team_idx: Index of team to check
            
        Returns:
            Tuple of (is_eliminated, certificate_teams)
        """
        # Check trivial elimination first
        if self.is_trivially_eliminated(team_idx):
            # Find teams that have already won too many games
            certificate = []
            max_wins = self.wins[team_idx] + self.remaining[team_idx]
            
            for i in range(self.n):
                if i != team_idx and self.wins[i] > max_wins:
                    certificate.append(self.teams[i])
                    
            return True, certificate
        
        # Build and solve flow network
        network, total_games = self.build_flow_network(team_idx)
        
        if total_games == 0:
            return False, None
            
        max_flow = network.ford_fulkerson("source", "sink")
        
        # Team is eliminated if max flow < total games
        if max_flow < total_games:
            # Find elimination certificate using min cut
            source_side = network.get_min_cut("source")
            certificate = []
            
            for i in range(self.n):
                if i != team_idx:
                    team_vertex = f"team_{i}"
                    if team_vertex in source_side:
                        certificate.append(self.teams[i])
                        
            return True, certificate
            
        return False, None
    
    def get_elimination_status(self) -> Dict[str, Tuple[bool, Optional[List[str]]]]:
        """
        Get elimination status for all teams.
        
        Returns:
            Dictionary mapping team names to (is_eliminated, certificate)
        """
        results = {}
        
        for i, team in enumerate(self.teams):
            is_elim, certificate = self.is_eliminated(i)
            results[team] = (is_elim, certificate)
            
        return results


class BaseballEliminationTester:
    """
    Generate test cases and analyze performance of the baseball elimination algorithm.
    """
    
    @staticmethod
    def generate_random_standings(n_teams: int, games_played_pct: float = 0.7,
                                 max_games: int = 162) -> Tuple[List[str], List[int], 
                                                                List[int], List[int], 
                                                                List[List[int]]]:
        """
        Generate random baseball standings for testing with consistent remaining games.
        
        Args:
            n_teams: Number of teams
            games_played_pct: Percentage of season completed
            max_games: Total games in season
            
        Returns:
            Tuple of (teams, wins, losses, remaining, games_matrix)
        """
        teams = [f"Team_{i}" for i in range(n_teams)]
        games_played = int(max_games * games_played_pct)
        
        # Generate wins/losses for games already played
        wins = []
        losses = []
        
        for i in range(n_teams):
            # Each team has played 'games_played' games
            w = random.randint(int(games_played * 0.3), int(games_played * 0.7))
            wins.append(w)
            losses.append(games_played - w)
            
        # Generate games matrix for remaining games
        # First create a matrix of remaining games
        games = [[0] * n_teams for _ in range(n_teams)]
        
        # Distribute remaining games between teams
        # In a real season, teams play each other roughly equally
        remaining_games_per_team = max_games - games_played
        games_per_matchup = remaining_games_per_team // (n_teams - 1)  # Rough estimate
        extra_games = remaining_games_per_team - games_per_matchup * (n_teams - 1)
        
        for i in range(n_teams):
            games_assigned = 0
            for j in range(n_teams):
                if i != j and i < j:  # Only fill upper triangle, then mirror
                    # Base games for this matchup
                    g = games_per_matchup
                    
                    # Add some randomness and extra games
                    if extra_games > 0 and random.random() < 0.3:
                        g += 1
                        extra_games -= 1
                    
                    # Add some variance
                    g = max(0, g + random.randint(-2, 2))
                    
                    games[i][j] = g
                    games[j][i] = g  # Symmetric
        
        # Calculate actual remaining games for each team
        remaining = []
        for i in range(n_teams):
            total = sum(games[i])
            remaining.append(total)
        
        return teams, wins, losses, remaining, games
    
    @staticmethod
    def create_elimination_scenario(n_teams: int) -> Tuple[List[str], List[int], 
                                                          List[int], List[int], 
                                                          List[List[int]]]:
        """
        Create a scenario where team 0 is definitely eliminated with consistent data.
        
        Args:
            n_teams: Number of teams
            
        Returns:
            Tuple of (teams, wins, losses, remaining, games_matrix)
        """
        teams = [f"Team_{i}" for i in range(n_teams)]
        
        # Team 0 has few wins, others have many
        wins = [30] + [80] * (n_teams - 1)
        losses = [80] + [30] * (n_teams - 1)
        
        # Create games matrix
        games = [[0] * n_teams for _ in range(n_teams)]
        
        # Each team plays some remaining games
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                if i == 0:
                    # Team 0 has fewer remaining games
                    games[i][j] = 1
                    games[j][i] = 1
                else:
                    # Other teams play each other more
                    games[i][j] = 2
                    games[j][i] = 2
        
        # Calculate actual remaining games (sum of each row)
        remaining = [sum(games[i]) for i in range(n_teams)]
                
        return teams, wins, losses, remaining, games
    
    @staticmethod
    def create_valid_test_case(scenario: str = "competitive") -> Tuple[List[str], List[int], 
                                                                      List[int], List[int], 
                                                                      List[List[int]]]:
        """
        Create various valid test scenarios with properly matched remaining games.
        
        Args:
            scenario: Type of scenario ("competitive", "elimination", "trivial")
            
        Returns:
            Tuple of (teams, wins, losses, remaining, games_matrix)
        """
        if scenario == "competitive":
            # Close race scenario
            teams = ["Yankees", "Red Sox", "Blue Jays", "Rays", "Orioles"]
            wins = [88, 87, 86, 85, 70]
            losses = [60, 61, 62, 63, 78]
            
            # Remaining games matrix (must be symmetric)
            games = [
                [0, 3, 3, 3, 3],   # Yankees: 12 remaining
                [3, 0, 3, 3, 3],   # Red Sox: 12 remaining  
                [3, 3, 0, 3, 3],   # Blue Jays: 12 remaining
                [3, 3, 3, 0, 3],   # Rays: 12 remaining
                [3, 3, 3, 3, 0]    # Orioles: 12 remaining
            ]
            
        elif scenario == "elimination":
            # Network flow elimination needed
            teams = ["Detroit", "New York", "Toronto", "Boston", "Baltimore"]
            wins = [75, 77, 77, 78, 74]
            losses = [59, 57, 57, 56, 60]
            
            games = [
                [0, 3, 8, 7, 3],   # Detroit: 21 remaining
                [3, 0, 2, 7, 4],   # New York: 16 remaining
                [8, 2, 0, 0, 0],   # Toronto: 10 remaining
                [7, 7, 0, 0, 0],   # Boston: 14 remaining
                [3, 4, 0, 0, 0]    # Baltimore: 7 remaining
            ]
            
        elif scenario == "trivial":
            # Trivial elimination case
            teams = ["LastPlace", "First", "Second", "Third"]
            wins = [40, 90, 85, 80]
            losses = [100, 50, 55, 60]
            
            games = [
                [0, 2, 2, 2],      # LastPlace: 6 remaining
                [2, 0, 3, 3],      # First: 8 remaining
                [2, 3, 0, 3],      # Second: 8 remaining
                [2, 3, 3, 0]       # Third: 8 remaining
            ]
        
        # Calculate remaining from games matrix
        remaining = [sum(row) for row in games]
        
        return teams, wins, losses, remaining, games
    
    @staticmethod
    def run_performance_test(team_sizes: List[int], iterations: int = 5):
        """
        Test algorithm performance with different league sizes.
        
        Args:
            team_sizes: List of team counts to test
            iterations: Number of iterations per size
            
        Returns:
            Dictionary of timing results
        """
        results = {}
        
        print(f"\n{'Teams':<8} {'Avg Time (s)':<15} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
        print("-" * 65)
        
        for n_teams in team_sizes:
            times = []
            
            for _ in range(iterations):
                # Generate random standings with consistent data
                data = BaseballEliminationTester.generate_random_standings(n_teams, 0.75)
                teams, wins, losses, remaining, games = data
                
                # Time the elimination check
                start_time = time.time()
                
                try:
                    be = BaseballElimination(teams, wins, losses, remaining, games)
                    
                    # Check elimination for all teams
                    for i in range(n_teams):
                        be.is_eliminated(i)
                        
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                except AssertionError as e:
                    print(f"Warning: Data validation failed for {n_teams} teams: {e}")
                    continue
                    
            if times:  # Only if we have valid timing data
                results[n_teams] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': min(times),
                    'max': max(times)
                }
                
                print(f"{n_teams:<8} {results[n_teams]['mean']:<15.6f} "
                      f"{results[n_teams]['std']:<12.6f} "
                      f"{results[n_teams]['min']:<12.6f} "
                      f"{results[n_teams]['max']:<12.6f}")
                  
        return results


def main():
    """
    Main function to demonstrate the baseball elimination algorithm.
    """
    print("=" * 70)
    print(" " * 15 + "Baseball Elimination Algorithm - CS 5800")
    print("=" * 70)
    
    # Example 1: Small valid example with proper remaining games
    print("\n--- Example 1: Small League (4 teams) ---")
    print("This example uses the textbook's classic elimination scenario")
    
    teams = ["Atlanta", "Philadelphia", "New York", "Montreal"]
    wins = [83, 80, 78, 77]
    losses = [71, 79, 78, 82]
    games = [
        [0, 1, 6, 1],   # Atlanta: 8 remaining (1+6+1)
        [1, 0, 0, 2],   # Philadelphia: 3 remaining (1+0+2)
        [6, 0, 0, 0],   # New York: 6 remaining (6+0+0)
        [1, 2, 0, 0]    # Montreal: 3 remaining (1+2+0)
    ]
    remaining = [sum(row) for row in games]  # Calculate from games matrix
    
    print("\nCurrent Standings:")
    print(f"{'Team':<15} {'Wins':<6} {'Losses':<8} {'Remaining':<10} {'Max Possible':<12}")
    print("-" * 55)
    for i, team in enumerate(teams):
        max_possible = wins[i] + remaining[i]
        print(f"{team:<15} {wins[i]:<6} {losses[i]:<8} {remaining[i]:<10} {max_possible:<12}")
    
    print("\nGames Remaining Between Teams:")
    print(f"{'Team':<15}", end="")
    for team in teams:
        print(f"{team[:8]:<10}", end="")
    print()
    for i, team in enumerate(teams):
        print(f"{team:<15}", end="")
        for j in range(len(teams)):
            print(f"{games[i][j]:<10}", end="")
        print()
    
    be = BaseballElimination(teams, wins, losses, remaining, games)
    
    print("\nElimination Analysis:")
    print("-" * 55)
    results = be.get_elimination_status()
    
    for team, (is_elim, certificate) in results.items():
        if is_elim:
            print(f"✗ {team}: ELIMINATED")
            if certificate:
                print(f"  Certificate of elimination: {', '.join(certificate)}")
                print(f"  These teams will average enough wins to eliminate {team}")
        else:
            print(f"✓ {team}: Still in contention")
    
    # Example 2: Test different scenarios
    print("\n--- Example 2: Different Scenarios ---")
    
    scenarios = ["competitive", "elimination", "trivial"]
    
    for scenario in scenarios:
        print(f"\n{scenario.upper()} Scenario:")
        print("-" * 40)
        
        data = BaseballEliminationTester.create_valid_test_case(scenario)
        teams, wins, losses, remaining, games = data
        
        be = BaseballElimination(teams, wins, losses, remaining, games)
        
        # Show brief standings
        print(f"{'Team':<15} {'W-L':<10} {'GB':<6} {'Elim?':<10}")
        leader_wins = max(wins)
        
        for i, team in enumerate(teams):
            record = f"{wins[i]}-{losses[i]}"
            gb = (leader_wins - wins[i])
            is_elim, _ = be.is_eliminated(i)
            status = "ELIM" if is_elim else "ALIVE"
            print(f"{team:<15} {record:<10} {gb:<6.1f} {status:<10}")
    
    # Example 3: Performance testing
    print("\n--- Example 3: Performance Analysis ---")
    print("Testing algorithm scalability with different league sizes...")
    
    team_sizes = [4, 6, 8, 10, 15, 20, 25, 30]
    results = BaseballEliminationTester.run_performance_test(team_sizes, iterations=3)
    
    if results:
        # Analyze complexity
        print("\n" + "=" * 70)
        print("Performance Summary:")
        print("-" * 70)
        
        # Check if performance matches expected O(n^4) complexity
        if len(results) > 1:
            sizes = sorted(results.keys())
            
            print(f"Smallest league ({sizes[0]} teams): {results[sizes[0]]['mean']:.6f} seconds")
            print(f"Largest league ({sizes[-1]} teams): {results[sizes[-1]]['mean']:.6f} seconds")
            
            # Estimate complexity
            ratio = results[sizes[-1]]['mean'] / results[sizes[0]]['mean']
            expected_ratio = (sizes[-1] / sizes[0]) ** 4
            
            print(f"\nComplexity Analysis:")
            print(f"  Time ratio (largest/smallest): {ratio:.2f}")
            print(f"  Expected for O(n^4): {expected_ratio:.2f}")
            print(f"  Complexity appears to be: O(n^{'4' if ratio/expected_ratio > 0.5 and ratio/expected_ratio < 2 else '?'})")
    
    print("\n" + "=" * 70)
    print("Algorithm demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()