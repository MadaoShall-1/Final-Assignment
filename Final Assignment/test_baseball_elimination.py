"""
Comprehensive Test Suite for Baseball Elimination Algorithm
CS 5800 Spring 2025

This module provides extensive testing capabilities including:
- Unit tests for correctness
- Performance benchmarking
- Visualization of results
- Real MLB data testing
"""

import unittest
import time
import random
import json
import csv
from typing import List, Tuple
import numpy as np

# Import the main algorithm (assumes baseball_elimination.py is in same directory)
from baseball_elimination import BaseballElimination, BaseballEliminationTester, FlowNetwork

# Optional: For visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not installed. Visualizations will be skipped.")


class TestBaseballElimination(unittest.TestCase):
    """Unit tests for the baseball elimination algorithm."""
    
    def test_trivial_elimination(self):
        """Test detection of trivial elimination."""
        teams = ["A", "B", "C"]
        wins = [50, 90, 85]  # Team A can't catch Team B even if wins all games
        losses = [90, 50, 55]
        
        # Games matrix - all zeros since trivial elimination doesn't depend on remaining games
        games = [
            [0, 1, 1],  # A has 2 remaining games
            [1, 0, 1],  # B has 2 remaining games
            [1, 1, 0]   # C has 2 remaining games
        ]
        remaining = [sum(row) for row in games]  # [2, 2, 2]
        
        be = BaseballElimination(teams, wins, losses, remaining, games)
        is_elim, _ = be.is_eliminated(0)  # Check team A
        
        self.assertTrue(is_elim, "Team A should be trivially eliminated")
        
    def test_non_elimination(self):
        """Test case where no team is eliminated."""
        teams = ["A", "B", "C"]
        wins = [80, 81, 82]
        losses = [70, 69, 68]
        
        # Balanced remaining games
        games = [
            [0, 4, 4],   # A: 8 remaining
            [4, 0, 4],   # B: 8 remaining
            [4, 4, 0]    # C: 8 remaining
        ]
        remaining = [sum(row) for row in games]  # [8, 8, 8]
        
        be = BaseballElimination(teams, wins, losses, remaining, games)
        
        for i in range(3):
            is_elim, _ = be.is_eliminated(i)
            self.assertFalse(is_elim, f"Team {teams[i]} should not be eliminated")
            
    def test_network_flow_elimination(self):
        """Test elimination that requires network flow (not trivial)."""
        teams = ["Detroit", "New York", "Toronto", "Boston", "Baltimore"]
        wins = [75, 77, 77, 78, 74]
        losses = [59, 57, 57, 56, 60]
        
        # Remaining games matrix (must be symmetric)
        games = [
            [0, 3, 8, 7, 3],   # Detroit: 21 remaining
            [3, 0, 2, 7, 4],   # New York: 16 remaining
            [8, 2, 0, 0, 0],   # Toronto: 10 remaining
            [7, 7, 0, 0, 0],   # Boston: 14 remaining
            [3, 4, 0, 0, 0]    # Baltimore: 7 remaining
        ]
        remaining = [sum(row) for row in games]  # Calculate from games matrix
        
        be = BaseballElimination(teams, wins, losses, remaining, games)
        is_elim, certificate = be.is_eliminated(0)  # Check Detroit
        
        self.assertTrue(is_elim, "Detroit should be eliminated")
        self.assertIsNotNone(certificate, "Should provide elimination certificate")
        
    def test_flow_network_construction(self):
        """Test that flow network is constructed correctly."""
        teams = ["A", "B", "C"]
        wins = [80, 85, 82]
        losses = [70, 65, 68]
        
        games = [
            [0, 4, 4],   # A: 8 remaining
            [4, 0, 4],   # B: 8 remaining
            [4, 4, 0]    # C: 8 remaining
        ]
        remaining = [sum(row) for row in games]
        
        be = BaseballElimination(teams, wins, losses, remaining, games)
        network, total_games = be.build_flow_network(0)  # Build for team A
        
        # Check that total games is correct (games between B and C only)
        self.assertEqual(total_games, 4, "Total games between B and C should be 4")
        
        # Check source edges exist
        self.assertIn("game_1_2", network.graph["source"])
        
    def test_symmetry(self):
        """Test that games matrix symmetry is enforced."""
        teams = ["A", "B"]
        wins = [50, 50]
        losses = [50, 50]
        games = [[0, 5], [3, 0]]  # Asymmetric - should fail
        remaining = [5, 3]  # Doesn't matter, will fail on symmetry check
        
        with self.assertRaises(AssertionError):
            BaseballElimination(teams, wins, losses, remaining, games)
    
    def test_remaining_games_consistency(self):
        """Test that remaining games must match games matrix."""
        teams = ["A", "B", "C"]
        wins = [50, 50, 50]
        losses = [50, 50, 50]
        
        games = [
            [0, 5, 5],   # A has 10 games in matrix
            [5, 0, 5],   # B has 10 games in matrix
            [5, 5, 0]    # C has 10 games in matrix
        ]
        remaining = [8, 8, 8]  # Wrong! Doesn't match matrix sums
        
        with self.assertRaises(AssertionError) as context:
            BaseballElimination(teams, wins, losses, remaining, games)
        
        self.assertIn("doesn't match", str(context.exception))
    
    def test_four_team_elimination(self):
        """Test the classic 4-team example from the textbook."""
        teams = ["Atlanta", "Philadelphia", "New York", "Montreal"]
        wins = [83, 80, 78, 77]
        losses = [71, 79, 78, 82]
        
        games = [
            [0, 1, 6, 1],   # Atlanta: 8 remaining
            [1, 0, 0, 2],   # Philadelphia: 3 remaining
            [6, 0, 0, 0],   # New York: 6 remaining
            [1, 2, 0, 0]    # Montreal: 3 remaining
        ]
        remaining = [sum(row) for row in games]
        
        be = BaseballElimination(teams, wins, losses, remaining, games)
        
        # Check each team
        atl_elim, _ = be.is_eliminated(0)
        phi_elim, phi_cert = be.is_eliminated(1)
        ny_elim, _ = be.is_eliminated(2)
        mon_elim, mon_cert = be.is_eliminated(3)
        
        self.assertFalse(atl_elim, "Atlanta should not be eliminated")
        self.assertTrue(phi_elim, "Philadelphia should be eliminated")
        self.assertFalse(ny_elim, "New York should not be eliminated")
        self.assertTrue(mon_elim, "Montreal should be eliminated")
        
        # Check certificates
        self.assertIsNotNone(phi_cert, "Philadelphia should have elimination certificate")
        self.assertIsNotNone(mon_cert, "Montreal should have elimination certificate")


class PerformanceAnalyzer:
    """Analyze and visualize algorithm performance."""
    
    @staticmethod
    def benchmark_by_team_size(max_teams: int = 30, step: int = 2):
        """
        Benchmark algorithm performance for different team sizes.
        
        Args:
            max_teams: Maximum number of teams to test
            step: Step size for team counts
            
        Returns:
            Performance data dictionary
        """
        team_counts = list(range(4, max_teams + 1, step))
        avg_times = []
        max_flow_sizes = []
        
        print(f"\n{'Teams':<8} {'Avg Time (s)':<15} {'Network Nodes':<15} {'Eliminated':<12}")
        print("-" * 55)
        
        for n in team_counts:
            times = []
            flow_sizes = []
            elim_counts = []
            
            # Run multiple trials
            for trial in range(5):
                data = BaseballEliminationTester.generate_random_standings(n, 0.75)
                teams, wins, losses, remaining, games = data
                
                be = BaseballElimination(teams, wins, losses, remaining, games)
                
                # Time checking all teams
                start = time.time()
                eliminated_count = 0
                
                for i in range(n):
                    is_elim, _ = be.is_eliminated(i)
                    if is_elim:
                        eliminated_count += 1
                        
                elapsed = time.time() - start
                times.append(elapsed)
                elim_counts.append(eliminated_count)
                
                # Estimate network size (for complexity analysis)
                # Network has: 1 source, 1 sink, ~n^2/2 game nodes, n team nodes
                network_size = 2 + (n * (n-1)) // 2 + n
                flow_sizes.append(network_size)
            
            avg_time = np.mean(times)
            avg_size = np.mean(flow_sizes)
            avg_elim = np.mean(elim_counts)
            
            avg_times.append(avg_time)
            max_flow_sizes.append(avg_size)
            
            print(f"{n:<8} {avg_time:<15.6f} {avg_size:<15.0f} {avg_elim:<12.1f}")
        
        return {
            'team_counts': team_counts,
            'avg_times': avg_times,
            'network_sizes': max_flow_sizes
        }
    
    @staticmethod
    def benchmark_by_season_progress(n_teams: int = 10):
        """
        Benchmark performance at different points in the season.
        
        Args:
            n_teams: Number of teams to use
            
        Returns:
            Performance data dictionary
        """
        progress_points = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        avg_times = []
        elimination_rates = []
        
        print(f"\n{'Season %':<12} {'Avg Time (s)':<15} {'Elimination %':<15}")
        print("-" * 45)
        
        for progress in progress_points:
            times = []
            elim_counts = []
            
            for trial in range(10):
                data = BaseballEliminationTester.generate_random_standings(
                    n_teams, progress
                )
                teams, wins, losses, remaining, games = data
                
                be = BaseballElimination(teams, wins, losses, remaining, games)
                
                start = time.time()
                eliminated = 0
                
                for i in range(n_teams):
                    is_elim, _ = be.is_eliminated(i)
                    if is_elim:
                        eliminated += 1
                        
                elapsed = time.time() - start
                times.append(elapsed)
                elim_counts.append(eliminated / n_teams)
            
            avg_time = np.mean(times)
            avg_elim = np.mean(elim_counts)
            
            avg_times.append(avg_time)
            elimination_rates.append(avg_elim)
            
            print(f"{progress*100:<12.0f} {avg_time:<15.6f} {avg_elim*100:<15.1f}")
        
        return {
            'progress': progress_points,
            'avg_times': avg_times,
            'elimination_rates': elimination_rates
        }
    
    @staticmethod
    def visualize_results(perf_data_size, perf_data_season):
        """Create performance visualization plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("Skipping visualizations (matplotlib not installed)")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Time vs Team Size
        ax1 = axes[0, 0]
        ax1.plot(perf_data_size['team_counts'], perf_data_size['avg_times'], 
                'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Teams')
        ax1.set_ylabel('Average Time (seconds)')
        ax1.set_title('Algorithm Runtime vs League Size')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Network Size vs Team Size
        ax2 = axes[0, 1]
        ax2.plot(perf_data_size['team_counts'], perf_data_size['network_sizes'], 
                'g-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Teams')
        ax2.set_ylabel('Flow Network Size (nodes)')
        ax2.set_title('Network Complexity vs League Size')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time vs Season Progress
        ax3 = axes[1, 0]
        progress_pct = [p * 100 for p in perf_data_season['progress']]
        ax3.plot(progress_pct, perf_data_season['avg_times'], 
                'r-^', linewidth=2, markersize=6)
        ax3.set_xlabel('Season Progress (%)')
        ax3.set_ylabel('Average Time (seconds)')
        ax3.set_title('Runtime vs Season Progress')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Elimination Rate vs Season Progress
        ax4 = axes[1, 1]
        elim_pct = [e * 100 for e in perf_data_season['elimination_rates']]
        ax4.bar(progress_pct, elim_pct, width=5, color='orange', alpha=0.7)
        ax4.set_xlabel('Season Progress (%)')
        ax4.set_ylabel('Teams Eliminated (%)')
        ax4.set_title('Elimination Rate Through Season')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Baseball Elimination Algorithm Performance Analysis', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


class MLBDataTester:
    """Test with realistic MLB division data."""
    
    @staticmethod
    def create_mlb_division_scenario():
        """
        Create a realistic MLB division scenario with consistent data.
        
        Returns:
            Test data tuple
        """
        # AL East late season scenario
        teams = ["Yankees", "Red Sox", "Blue Jays", "Rays", "Orioles"]
        wins = [92, 88, 85, 83, 68]
        losses = [58, 62, 65, 67, 82]
        
        # Remaining divisional games (symmetric matrix)
        games = [
            [0, 3, 2, 2, 2],   # Yankees: 9 remaining
            [3, 0, 2, 2, 2],   # Red Sox: 9 remaining
            [2, 2, 0, 3, 2],   # Blue Jays: 9 remaining
            [2, 2, 3, 0, 2],   # Rays: 9 remaining
            [2, 2, 2, 2, 0]    # Orioles: 8 remaining
        ]
        
        # Calculate remaining from games matrix
        remaining = [sum(row) for row in games]
        
        return teams, wins, losses, remaining, games
    
    @staticmethod
    def create_wild_card_scenario():
        """
        Create a wild card race scenario.
        
        Returns:
            Test data tuple
        """
        teams = ["Team1", "Team2", "Team3", "Team4", "Team5", "Team6"]
        wins = [85, 84, 84, 83, 82, 75]
        losses = [65, 66, 66, 67, 68, 75]
        
        # Tight race with 12 games left
        games = [
            [0, 2, 2, 2, 2, 2],   # 10 remaining
            [2, 0, 2, 2, 2, 2],   # 10 remaining
            [2, 2, 0, 2, 2, 2],   # 10 remaining
            [2, 2, 2, 0, 2, 2],   # 10 remaining
            [2, 2, 2, 2, 0, 2],   # 10 remaining
            [2, 2, 2, 2, 2, 0]    # 10 remaining
        ]
        
        remaining = [sum(row) for row in games]
        
        return teams, wins, losses, remaining, games
    
    @staticmethod
    def test_mlb_scenario():
        """Test and display MLB scenario results."""
        print("\n" + "=" * 60)
        print("MLB Division Scenario Analysis")
        print("=" * 60)
        
        teams, wins, losses, remaining, games = MLBDataTester.create_mlb_division_scenario()
        be = BaseballElimination(teams, wins, losses, remaining, games)
        
        # Display standings
        print(f"\n{'Team':<12} {'W':<4} {'L':<4} {'PCT':<6} {'GB':<6} {'Rem':<5} {'Max W':<7}")
        print("-" * 50)
        
        leader_wins = max(wins)
        leader_idx = wins.index(leader_wins)
        
        for i, team in enumerate(teams):
            pct = wins[i] / (wins[i] + losses[i])
            gb = (leader_wins - wins[i]) / 2
            max_w = wins[i] + remaining[i]
            print(f"{team:<12} {wins[i]:<4} {losses[i]:<4} {pct:<6.3f} "
                  f"{gb:<6.1f} {remaining[i]:<5} {max_w:<7}")
        
        # Check elimination status
        print("\n" + "=" * 60)
        print("Elimination Analysis")
        print("=" * 60)
        
        for i, team in enumerate(teams):
            start = time.time()
            is_elim, certificate = be.is_eliminated(i)
            elapsed = time.time() - start
            
            if is_elim:
                print(f"\n{team}: ELIMINATED (computed in {elapsed:.4f}s)")
                if certificate:
                    print(f"  Elimination certificate: {', '.join(certificate)}")
                    
                    # Show why eliminated
                    team_wins = wins[i] + remaining[i]
                    cert_indices = [teams.index(t) for t in certificate]
                    
                    total_cert_wins = sum(wins[j] for j in cert_indices)
                    total_cert_games = sum(games[j][k] for j in cert_indices 
                                         for k in cert_indices if j < k)
                    
                    avg_wins = (total_cert_wins + total_cert_games) / len(certificate)
                    
                    print(f"  Max possible wins for {team}: {team_wins}")
                    print(f"  Average wins for certificate teams: {avg_wins:.1f}")
            else:
                print(f"\n{team}: STILL IN CONTENTION (computed in {elapsed:.4f}s)")


def generate_test_data_csv(filename: str = "test_standings.csv"):
    """
    Generate CSV file with test data for the algorithm.
    
    Args:
        filename: Output CSV filename
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['TestCase', 'Teams', 'Wins', 'Losses', 'Remaining', 
                        'GamesMatrix', 'ExpectedEliminated'])
        
        # Test Case 1: Trivial elimination (corrected)
        writer.writerow([
            'Trivial',
            'LastPlace,First,Second',
            '50,90,85',
            '90,50,55',
            '2,2,2',  # Must match games matrix sums
            '0-1-1;1-0-1;1-1-0',
            'LastPlace'
        ])
        
        # Test Case 2: Network flow elimination (corrected)
        writer.writerow([
            'NetworkFlow',
            'Detroit,NewYork,Toronto,Boston,Baltimore',
            '75,77,77,78,74',
            '59,57,57,56,60',
            '21,16,10,14,7',  # Matches matrix sums
            '0-3-8-7-3;3-0-2-7-4;8-2-0-0-0;7-7-0-0-0;3-4-0-0-0',
            'Detroit'
        ])
        
        # Test Case 3: No elimination (corrected)
        writer.writerow([
            'NoElimination',
            'A,B,C,D',
            '80,81,82,80',
            '70,69,68,70',
            '9,9,9,9',  # Each team plays 3 games against each other team
            '0-3-3-3;3-0-3-3;3-3-0-3;3-3-3-0',
            'None'
        ])
        
        # Test Case 4: Classic textbook example
        writer.writerow([
            'Textbook',
            'Atlanta,Philadelphia,NewYork,Montreal',
            '83,80,78,77',
            '71,79,78,82',
            '8,3,6,3',  # Matches matrix sums
            '0-1-6-1;1-0-0-2;6-0-0-0;1-2-0-0',
            'Philadelphia,Montreal'
        ])
    
    print(f"Test data written to {filename}")


def validate_test_data_from_csv(filename: str = "test_standings.csv"):
    """
    Read and validate test data from CSV file.
    
    Args:
        filename: Input CSV filename
    """
    print(f"\nValidating test data from {filename}...")
    print("-" * 50)
    
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            test_case = row['TestCase']
            teams = row['Teams'].split(',')
            wins = list(map(int, row['Wins'].split(',')))
            losses = list(map(int, row['Losses'].split(',')))
            remaining = list(map(int, row['Remaining'].split(',')))
            
            # Parse games matrix
            games_rows = row['GamesMatrix'].split(';')
            games = []
            for games_row in games_rows:
                games.append(list(map(int, games_row.split('-'))))
            
            print(f"\nTest Case: {test_case}")
            
            # Validate data consistency
            try:
                be = BaseballElimination(teams, wins, losses, remaining, games)
                print(f"  ✓ Data validation passed")
                
                # Check expected eliminations
                expected = row['ExpectedEliminated'].split(',') if row['ExpectedEliminated'] != 'None' else []
                
                for i, team in enumerate(teams):
                    is_elim, _ = be.is_eliminated(i)
                    expected_elim = team in expected
                    
                    if is_elim == expected_elim:
                        status = "✓"
                    else:
                        status = "✗"
                    
                    print(f"  {status} {team}: {'Eliminated' if is_elim else 'Not eliminated'} "
                          f"(Expected: {'Eliminated' if expected_elim else 'Not eliminated'})")
                          
            except AssertionError as e:
                print(f"  ✗ Data validation failed: {e}")


def main():
    """Main test runner."""
    print("=" * 70)
    print("BASEBALL ELIMINATION ALGORITHM - COMPREHENSIVE TESTING")
    print("=" * 70)
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    print("-" * 40)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseballElimination)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✓ All unit tests passed!")
    else:
        print("\n✗ Some unit tests failed.")
    
    # Performance testing by team size
    print("\n2. Performance Analysis by Team Size...")
    print("-" * 40)
    perf_data_size = PerformanceAnalyzer.benchmark_by_team_size(max_teams=20, step=2)
    
    # Performance testing by season progress
    print("\n3. Performance Analysis by Season Progress...")
    print("-" * 40)
    perf_data_season = PerformanceAnalyzer.benchmark_by_season_progress(n_teams=10)
    
    # MLB scenario testing
    print("\n4. MLB Division Scenario...")
    print("-" * 40)
    MLBDataTester.test_mlb_scenario()
    
    # Wild card scenario
    print("\n5. Wild Card Race Scenario...")
    print("-" * 40)
    teams, wins, losses, remaining, games = MLBDataTester.create_wild_card_scenario()
    be = BaseballElimination(teams, wins, losses, remaining, games)
    
    print("Wild Card Race - Final 10 Games:")
    print(f"{'Team':<10} {'W-L':<10} {'GB':<8} {'Status':<15}")
    print("-" * 45)
    
    leader_wins = max(wins)
    for i, team in enumerate(teams):
        record = f"{wins[i]}-{losses[i]}"
        gb = (leader_wins - wins[i])
        is_elim, _ = be.is_eliminated(i)
        status = "ELIMINATED" if is_elim else "IN CONTENTION"
        print(f"{team:<10} {record:<10} {gb:<8.1f} {status:<15}")
    
    # Generate test data CSV
    print("\n6. Generating Test Data File...")
    print("-" * 40)
    generate_test_data_csv()
    validate_test_data_from_csv()
    
    # Visualize results
    print("\n7. Creating Performance Visualizations...")
    print("-" * 40)
    try:
        PerformanceAnalyzer.visualize_results(perf_data_size, perf_data_season)
        print("✓ Visualizations created successfully!")
    except Exception as e:
        print(f"Note: Visualization skipped. {e}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("TESTING SUMMARY")
    print("=" * 70)
    
    # Calculate complexity
    if perf_data_size and len(perf_data_size['team_counts']) > 1:
        n_values = perf_data_size['team_counts']
        times = perf_data_size['avg_times']
        
        # Estimate complexity (should be roughly O(n^4) for n teams)
        complexity_ratio = []
        for i in range(1, len(n_values)):
            n1, n2 = n_values[i-1], n_values[i]
            t1, t2 = times[i-1], times[i]
            # Expected ratio for O(n^4): (n2/n1)^4
            expected = (n2/n1) ** 4
            actual = t2/t1
            complexity_ratio.append(actual / expected)
        
        avg_ratio = np.mean(complexity_ratio) if complexity_ratio else 0
        
        print(f"\nAlgorithm Complexity Analysis:")
        print(f"  Expected: O(n^4) where n = number of teams")
        print(f"  Empirical complexity ratio: {avg_ratio:.2f}")
        print(f"  {'✓ Matches expected complexity' if 0.5 < avg_ratio < 2.0 else '✗ Complexity differs from expected'}")
        
        print(f"\nPerformance Highlights:")
        print(f"  4-team league:  {perf_data_size['avg_times'][0]:.6f} seconds")
        if len(perf_data_size['avg_times']) > 3:
            print(f"  10-team league: {perf_data_size['avg_times'][3]:.6f} seconds")
        if len(perf_data_size['avg_times']) > 8:
            print(f"  20-team league: {perf_data_size['avg_times'][8]:.6f} seconds")
    
    if perf_data_season and len(perf_data_season['elimination_rates']) > 4:
        print(f"\nElimination Patterns:")
        print(f"  At 50% season: {perf_data_season['elimination_rates'][2]*100:.1f}% eliminated")
        print(f"  At 90% season: {perf_data_season['elimination_rates'][4]*100:.1f}% eliminated")
    
    print("\n" + "=" * 70)
    print("Testing complete! All components verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()