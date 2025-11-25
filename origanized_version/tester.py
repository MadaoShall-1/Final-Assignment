"""
Comprehensive Test Suite for Baseball Elimination Algorithm
CS 5800 Spring 2025

Includes: Test data generation, unit tests, performance benchmarks, 
visualization, and CSV testing.
"""

import unittest
import time
import csv
import random
from typing import List, Tuple
import numpy as np

from baseball_elimination import BaseballElimination

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class TestDataGenerator:
    """Generate test data for baseball elimination algorithm."""
    
    @staticmethod
    def generate_random_standings(n_teams: int, games_played_pct: float = 0.7,
                                 max_games: int = 162) -> Tuple[List[str], List[int], 
                                                                List[int], List[int], 
                                                                List[List[int]]]:
        """Generate random baseball standings for testing."""
        teams = [f"Team_{i}" for i in range(n_teams)]
        games_played = int(max_games * games_played_pct)
        
        wins, losses = [], []
        for _ in range(n_teams):
            w = random.randint(int(games_played * 0.3), int(games_played * 0.7))
            wins.append(w)
            losses.append(games_played - w)
        
        games = [[0] * n_teams for _ in range(n_teams)]
        remaining_per_team = max_games - games_played
        games_per_matchup = remaining_per_team // (n_teams - 1)
        
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                g = max(0, games_per_matchup + random.randint(-2, 2))
                games[i][j] = games[j][i] = g
        
        remaining = [sum(games[i]) for i in range(n_teams)]
        return teams, wins, losses, remaining, games
    
    @staticmethod
    def create_scenario(name: str) -> Tuple[List[str], List[int], 
                                            List[int], List[int], 
                                            List[List[int]]]:
        """
        Create predefined test scenarios.
        
        Args:
            name: One of 'competitive', 'elimination', 'trivial', 
                  'textbook', 'mlb_division', 'wild_card'
        """
        scenarios = {
            "competitive": {
                "teams": ["Yankees", "Red Sox", "Blue Jays", "Rays", "Orioles"],
                "wins": [88, 87, 86, 85, 70],
                "losses": [60, 61, 62, 63, 78],
                "games": [[0,3,3,3,3], [3,0,3,3,3], [3,3,0,3,3], 
                         [3,3,3,0,3], [3,3,3,3,0]]
            },
            "elimination": {
                "teams": ["Detroit", "New York", "Toronto", "Boston", "Baltimore"],
                "wins": [75, 77, 77, 78, 74],
                "losses": [59, 57, 57, 56, 60],
                "games": [[0,3,8,7,3], [3,0,2,7,4], [8,2,0,0,0], 
                         [7,7,0,0,0], [3,4,0,0,0]]
            },
            "trivial": {
                "teams": ["LastPlace", "First", "Second", "Third"],
                "wins": [40, 90, 85, 80],
                "losses": [100, 50, 55, 60],
                "games": [[0,2,2,2], [2,0,3,3], [2,3,0,3], [2,3,3,0]]
            },
            "textbook": {
                "teams": ["Atlanta", "Philadelphia", "New York", "Montreal"],
                "wins": [83, 80, 78, 77],
                "losses": [71, 79, 78, 82],
                "games": [[0,1,6,1], [1,0,0,2], [6,0,0,0], [1,2,0,0]]
            },
            "mlb_division": {
                "teams": ["Yankees", "Red Sox", "Blue Jays", "Rays", "Orioles"],
                "wins": [92, 88, 85, 83, 68],
                "losses": [58, 62, 65, 67, 82],
                "games": [[0,3,2,2,2], [3,0,2,2,2], [2,2,0,3,2], 
                         [2,2,3,0,2], [2,2,2,2,0]]
            },
            "wild_card": {
                "teams": ["Team1", "Team2", "Team3", "Team4", "Team5", "Team6"],
                "wins": [85, 84, 84, 83, 82, 75],
                "losses": [65, 66, 66, 67, 68, 75],
                "games": [[0,2,2,2,2,2], [2,0,2,2,2,2], [2,2,0,2,2,2],
                         [2,2,2,0,2,2], [2,2,2,2,0,2], [2,2,2,2,2,0]]
            }
        }
        
        s = scenarios.get(name, scenarios["competitive"])
        remaining = [sum(row) for row in s["games"]]
        return s["teams"], s["wins"], s["losses"], remaining, s["games"]
    
    @staticmethod
    def create_elimination_scenario(n_teams: int):
        """Create scenario where team 0 is eliminated."""
        teams = [f"Team_{i}" for i in range(n_teams)]
        wins = [30] + [80] * (n_teams - 1)
        losses = [80] + [30] * (n_teams - 1)
        
        games = [[0] * n_teams for _ in range(n_teams)]
        for i in range(n_teams):
            for j in range(i + 1, n_teams):
                g = 1 if i == 0 else 2
                games[i][j] = games[j][i] = g
        
        remaining = [sum(games[i]) for i in range(n_teams)]
        return teams, wins, losses, remaining, games


# Backward compatibility alias
BaseballEliminationTester = TestDataGenerator


class TestBaseballElimination(unittest.TestCase):
    """Unit tests for the baseball elimination algorithm."""
    
    def test_trivial_elimination(self):
        """Test detection of trivial elimination."""
        data = TestDataGenerator.create_scenario("trivial")
        be = BaseballElimination(*data)
        is_elim, _ = be.is_eliminated(0)
        self.assertTrue(is_elim, "LastPlace should be trivially eliminated")
        
    def test_non_elimination(self):
        """Test case where no team is eliminated."""
        teams = ["A", "B", "C"]
        wins, losses = [80, 81, 82], [70, 69, 68]
        games = [[0,4,4], [4,0,4], [4,4,0]]
        remaining = [8, 8, 8]
        
        be = BaseballElimination(teams, wins, losses, remaining, games)
        for i in range(3):
            is_elim, _ = be.is_eliminated(i)
            self.assertFalse(is_elim, f"Team {teams[i]} should not be eliminated")
            
    def test_network_flow_elimination(self):
        """Test elimination requiring network flow (not trivial)."""
        data = TestDataGenerator.create_scenario("elimination")
        be = BaseballElimination(*data)
        is_elim, cert = be.is_eliminated(0)
        
        self.assertTrue(is_elim, "Detroit should be eliminated")
        self.assertIsNotNone(cert, "Should provide certificate")
        
    def test_flow_network_construction(self):
        """Test flow network is constructed correctly."""
        teams = ["A", "B", "C"]
        wins, losses = [80, 85, 82], [70, 65, 68]
        games = [[0,4,4], [4,0,4], [4,4,0]]
        remaining = [8, 8, 8]
        
        be = BaseballElimination(teams, wins, losses, remaining, games)
        network, total = be.build_flow_network(0)
        
        self.assertEqual(total, 4, "Games between B and C should be 4")
        self.assertIn("game_1_2", network.graph["source"])
        
    def test_symmetry_validation(self):
        """Test that asymmetric games matrix is rejected."""
        teams, wins, losses = ["A", "B"], [50, 50], [50, 50]
        games = [[0, 5], [3, 0]]
        
        with self.assertRaises(AssertionError):
            BaseballElimination(teams, wins, losses, [5, 3], games)
    
    def test_remaining_games_consistency(self):
        """Test remaining games must match matrix."""
        teams = ["A", "B", "C"]
        wins, losses = [50, 50, 50], [50, 50, 50]
        games = [[0,5,5], [5,0,5], [5,5,0]]
        
        with self.assertRaises(AssertionError):
            BaseballElimination(teams, wins, losses, [8, 8, 8], games)
    
    def test_textbook_example(self):
        """Test classic 4-team textbook example."""
        data = TestDataGenerator.create_scenario("textbook")
        be = BaseballElimination(*data)
        
        self.assertFalse(be.is_eliminated(0)[0], "Atlanta not eliminated")
        self.assertTrue(be.is_eliminated(1)[0], "Philadelphia eliminated")
        self.assertFalse(be.is_eliminated(2)[0], "New York not eliminated")
        self.assertTrue(be.is_eliminated(3)[0], "Montreal eliminated")


class PerformanceAnalyzer:
    """Analyze algorithm performance."""
    
    @staticmethod
    def benchmark_by_team_size(max_teams: int = 30, step: int = 2):
        """Benchmark for different team sizes."""
        results = {'team_counts': [], 'avg_times': [], 'network_sizes': []}
        
        print(f"\n{'Teams':<8} {'Avg Time (s)':<15} {'Network Nodes':<15}")
        print("-" * 40)
        
        for n in range(4, max_teams + 1, step):
            times = []
            for _ in range(5):
                data = TestDataGenerator.generate_random_standings(n, 0.75)
                be = BaseballElimination(*data)
                
                start = time.time()
                for i in range(n):
                    be.is_eliminated(i)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            net_size = 2 + (n * (n-1)) // 2 + n
            
            results['team_counts'].append(n)
            results['avg_times'].append(avg_time)
            results['network_sizes'].append(net_size)
            
            print(f"{n:<8} {avg_time:<15.6f} {net_size:<15.0f}")
        
        return results
    
    @staticmethod
    def benchmark_by_season_progress(n_teams: int = 10):
        """Benchmark at different season points."""
        results = {'progress': [], 'avg_times': [], 'elimination_rates': []}
        
        print(f"\n{'Season %':<12} {'Avg Time (s)':<15} {'Elim %':<15}")
        print("-" * 45)
        
        for pct in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            times, elim_rates = [], []
            
            for _ in range(10):
                data = TestDataGenerator.generate_random_standings(n_teams, pct)
                be = BaseballElimination(*data)
                
                start = time.time()
                elim = sum(1 for i in range(n_teams) if be.is_eliminated(i)[0])
                times.append(time.time() - start)
                elim_rates.append(elim / n_teams)
            
            results['progress'].append(pct)
            results['avg_times'].append(np.mean(times))
            results['elimination_rates'].append(np.mean(elim_rates))
            
            print(f"{pct*100:<12.0f} {np.mean(times):<15.6f} {np.mean(elim_rates)*100:<15.1f}")
        
        return results
    
    @staticmethod
    def visualize_results(size_data, season_data):
        """Create performance plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not installed, skipping visualization")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].plot(size_data['team_counts'], size_data['avg_times'], 'b-o')
        axes[0,0].set_xlabel('Teams'); axes[0,0].set_ylabel('Time (s)')
        axes[0,0].set_title('Runtime vs League Size'); axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(size_data['team_counts'], size_data['network_sizes'], 'g-s')
        axes[0,1].set_xlabel('Teams'); axes[0,1].set_ylabel('Nodes')
        axes[0,1].set_title('Network Size vs League Size'); axes[0,1].grid(True, alpha=0.3)
        
        pct = [p*100 for p in season_data['progress']]
        axes[1,0].plot(pct, season_data['avg_times'], 'r-^')
        axes[1,0].set_xlabel('Season %'); axes[1,0].set_ylabel('Time (s)')
        axes[1,0].set_title('Runtime vs Season Progress'); axes[1,0].grid(True, alpha=0.3)
        
        axes[1,1].bar(pct, [e*100 for e in season_data['elimination_rates']], 
                      width=5, color='orange', alpha=0.7)
        axes[1,1].set_xlabel('Season %'); axes[1,1].set_ylabel('Eliminated %')
        axes[1,1].set_title('Elimination Rate'); axes[1,1].grid(True, alpha=0.3)
        
        plt.suptitle('Baseball Elimination Performance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


class CSVTestManager:
    """Manage CSV-based test data."""
    
    @staticmethod
    def generate_csv(filename: str = "test_standings.csv"):
        """Generate test data CSV file."""
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['TestCase', 'Teams', 'Wins', 'Losses', 'Remaining', 
                       'GamesMatrix', 'ExpectedEliminated'])
            
            w.writerow(['Trivial', 'LastPlace,First,Second', '50,90,85', 
                       '90,50,55', '2,2,2', '0-1-1;1-0-1;1-1-0', 'LastPlace'])
            w.writerow(['NetworkFlow', 'Detroit,NewYork,Toronto,Boston,Baltimore',
                       '75,77,77,78,74', '59,57,57,56,60', '21,16,10,14,7',
                       '0-3-8-7-3;3-0-2-7-4;8-2-0-0-0;7-7-0-0-0;3-4-0-0-0', 'Detroit'])
            w.writerow(['Textbook', 'Atlanta,Philadelphia,NewYork,Montreal',
                       '83,80,78,77', '71,79,78,82', '8,3,6,3',
                       '0-1-6-1;1-0-0-2;6-0-0-0;1-2-0-0', 'Philadelphia,Montreal'])
        print(f"Generated {filename}")
    
    @staticmethod
    def validate_csv(filename: str = "test_standings.csv"):
        """Validate test data from CSV."""
        print(f"\nValidating {filename}...")
        
        with open(filename, 'r') as f:
            for row in csv.DictReader(f):
                teams = row['Teams'].split(',')
                wins = list(map(int, row['Wins'].split(',')))
                losses = list(map(int, row['Losses'].split(',')))
                remaining = list(map(int, row['Remaining'].split(',')))
                games = [list(map(int, r.split('-'))) for r in row['GamesMatrix'].split(';')]
                expected = row['ExpectedEliminated'].split(',') if row['ExpectedEliminated'] != 'None' else []
                
                try:
                    be = BaseballElimination(teams, wins, losses, remaining, games)
                    print(f"\n{row['TestCase']}: ✓ Valid")
                    
                    for i, team in enumerate(teams):
                        is_elim = be.is_eliminated(i)[0]
                        exp = team in expected
                        status = "✓" if is_elim == exp else "✗"
                        print(f"  {status} {team}: {'Elim' if is_elim else 'Alive'}")
                except AssertionError as e:
                    print(f"\n{row['TestCase']}: ✗ Invalid - {e}")


def run_all_tests():
    """Run complete test suite."""
    print("=" * 70)
    print("BASEBALL ELIMINATION - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    # Unit tests
    print("\n1. Unit Tests")
    print("-" * 40)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseballElimination)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Performance by size
    print("\n2. Performance by Team Size")
    print("-" * 40)
    size_data = PerformanceAnalyzer.benchmark_by_team_size(max_teams=20, step=2)
    
    # Performance by season
    print("\n3. Performance by Season Progress")
    print("-" * 40)
    season_data = PerformanceAnalyzer.benchmark_by_season_progress(n_teams=10)
    
    # MLB scenarios
    print("\n4. MLB Scenarios")
    print("-" * 40)
    for scenario in ["mlb_division", "wild_card"]:
        data = TestDataGenerator.create_scenario(scenario)
        be = BaseballElimination(*data)
        
        print(f"\n{scenario.upper()}:")
        print(f"{'Team':<12} {'W-L':<10} {'Status':<12}")
        for i, team in enumerate(data[0]):
            is_elim = be.is_eliminated(i)[0]
            print(f"{team:<12} {data[1][i]}-{data[2][i]:<6} {'ELIM' if is_elim else 'ALIVE':<12}")
    
    # CSV tests
    print("\n5. CSV Test Data")
    print("-" * 40)
    CSVTestManager.generate_csv()
    CSVTestManager.validate_csv()
    
    # Visualization
    print("\n6. Visualization")
    print("-" * 40)
    PerformanceAnalyzer.visualize_results(size_data, season_data)
    
    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
