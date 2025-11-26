"""
Baseball Elimination Algorithm
Solves the baseball elimination problem using network flow.
"""

from typing import List, Dict, Tuple, Optional
from flow_network import FlowNetwork


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
        self._validate_input()
        
    def _validate_input(self):
        """Validate that input data is consistent."""
        assert len(self.teams) == len(self.wins) == len(self.losses) == len(self.remaining)
        assert len(self.games) == self.n
        
        for i in range(self.n):
            assert len(self.games[i]) == self.n
            assert self.games[i][i] == 0
            
        for i in range(self.n):
            for j in range(i + 1, self.n):
                assert self.games[i][j] == self.games[j][i], \
                    f"Games matrix not symmetric at [{i}][{j}]"
        
        for i in range(self.n):
            total = sum(self.games[i])
            assert total == self.remaining[i], \
                f"Team {self.teams[i]}: remaining mismatch"
                
    def is_trivially_eliminated(self, team_idx: int) -> bool:
        """Check if a team is trivially eliminated."""
        max_possible = self.wins[team_idx] + self.remaining[team_idx]
        return any(i != team_idx and self.wins[i] > max_possible 
                   for i in range(self.n))
    
    def build_flow_network(self, team_idx: int) -> Tuple[FlowNetwork, int]:
        """Build flow network to check if team is eliminated."""
        network = FlowNetwork()
        max_wins = self.wins[team_idx] + self.remaining[team_idx]
        total_games = 0
        
        for i in range(self.n):
            if i == team_idx:
                continue
            for j in range(i + 1, self.n):
                if j == team_idx:
                    continue
                if self.games[i][j] > 0:
                    game_v = f"game_{i}_{j}"
                    network.add_edge("source", game_v, self.games[i][j])
                    total_games += self.games[i][j]
                    network.add_edge(game_v, f"team_{i}", float('inf'))
                    network.add_edge(game_v, f"team_{j}", float('inf'))
        
        for i in range(self.n):
            if i != team_idx:
                cap = max(0, max_wins - self.wins[i])
                network.add_edge(f"team_{i}", "sink", cap)
            
        return network, total_games
    
    def is_eliminated(self, team_idx: int) -> Tuple[bool, Optional[List[str]]]:
        """Check if a team is eliminated and find certificate if so."""
        if self.is_trivially_eliminated(team_idx):
            max_wins = self.wins[team_idx] + self.remaining[team_idx]
            cert = [self.teams[i] for i in range(self.n) 
                    if i != team_idx and self.wins[i] > max_wins]
            return True, cert
        
        network, total_games = self.build_flow_network(team_idx)
        
        if total_games == 0:
            return False, None
            
        max_flow = network.ford_fulkerson("source", "sink")
        
        if max_flow < total_games:
            source_side = network.get_min_cut("source")
            cert = [self.teams[i] for i in range(self.n)
                    if i != team_idx and f"team_{i}" in source_side]
            return True, cert
            
        return False, None
    
    def get_elimination_status(self) -> Dict[str, Tuple[bool, Optional[List[str]]]]:
        """Get elimination status for all teams."""
        return {team: self.is_eliminated(i) for i, team in enumerate(self.teams)}
