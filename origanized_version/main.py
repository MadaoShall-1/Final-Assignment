"""
Baseball Elimination Algorithm - Main Entry Point
CS 5800 Spring 2025 Final Project
"""

from baseball_elimination import BaseballElimination
from tester import BaseballEliminationTester
from visualization import visualize_elimination_network


def main():
    """Demonstrate the baseball elimination algorithm."""
    print("=" * 70)
    print(" " * 15 + "Baseball Elimination Algorithm - CS 5800")
    print("=" * 70)
    
    # Example 1: Classic textbook example
    print("\n--- Example 1: Small League (4 teams) ---")
    
    teams = ["Atlanta", "Philadelphia", "New York", "Montreal"]
    wins = [83, 80, 78, 77]
    losses = [71, 79, 78, 82]
    games = [[0,1,6,1], [1,0,0,2], [6,0,0,0], [1,2,0,0]]
    remaining = [sum(row) for row in games]
    
    print("\nCurrent Standings:")
    print(f"{'Team':<15} {'Wins':<6} {'Losses':<8} {'Remaining':<10}")
    print("-" * 45)
    for i, team in enumerate(teams):
        print(f"{team:<15} {wins[i]:<6} {losses[i]:<8} {remaining[i]:<10}")
    
    print("\nGames Remaining Between Teams:")
    print(f"{'Team':<15}", end="")
    for t in teams:
        print(f"{t[:8]:<10}", end="")
    print()
    for i, team in enumerate(teams):
        print(f"{team:<15}", end="")
        for j in range(len(teams)):
            print(f"{games[i][j]:<10}", end="")
        print()
    
    be = BaseballElimination(teams, wins, losses, remaining, games)
    
    print("\nElimination Analysis:")
    print("-" * 45)
    for team, (is_elim, cert) in be.get_elimination_status().items():
        if is_elim:
            print(f"✗ {team}: ELIMINATED")
            if cert:
                print(f"  Certificate: {', '.join(cert)}")
        else:
            print(f"✓ {team}: Still in contention")
    
    # Example 2: Visualize network for Philadelphia
    print("\n--- Example 2: Network Visualization ---")
    print("Generating flow network visualization for Philadelphia...")
    visualize_elimination_network(be, team_idx=1, show_flow=True)
    
    # Example 3: Different scenarios
    print("\n--- Example 3: Different Scenarios ---")
    for scenario in ["competitive", "elimination", "trivial"]:
        print(f"\n{scenario.upper()} Scenario:")
        print("-" * 40)
        
        data = BaseballEliminationTester.create_valid_test_case(scenario)
        be = BaseballElimination(*data)
        leader_wins = max(data[1])
        
        print(f"{'Team':<15} {'W-L':<10} {'GB':<6} {'Status':<10}")
        for i, team in enumerate(data[0]):
            record = f"{data[1][i]}-{data[2][i]}"
            gb = leader_wins - data[1][i]
            is_elim, _ = be.is_eliminated(i)
            print(f"{team:<15} {record:<10} {gb:<6.1f} {'ELIM' if is_elim else 'ALIVE':<10}")
    
    # Example 4: Performance testing
    print("\n--- Example 4: Performance Analysis ---")
    BaseballEliminationTester.run_performance_test([4, 6, 8, 10, 15, 20], iterations=3)
    
    print("\n" + "=" * 70)
    print("Algorithm demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
