"""
Interactive Data Loader for Baseball Elimination
Supports manual input, CSV files, and JSON files.
"""

import csv
import json
from typing import Tuple, List
from baseball_elimination import BaseballElimination


def load_from_csv(standings_file: str, games_file: str) -> BaseballElimination:
    """
    Load data from two CSV files.
    
    standings_file format:
        Team,Wins,Losses
        Yankees,92,58
        ...
    
    games_file format (matrix with headers):
        ,Yankees,Red Sox,...
        Yankees,0,3,...
        Red Sox,3,0,...
    """
    teams, wins, losses = [], [], []
    with open(standings_file, 'r') as f:
        for row in csv.DictReader(f):
            teams.append(row['Team'])
            wins.append(int(row['Wins']))
            losses.append(int(row['Losses']))
    
    games = []
    with open(games_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            games.append([int(x) for x in row[1:]])
    
    remaining = [sum(row) for row in games]
    return BaseballElimination(teams, wins, losses, remaining, games)


def load_from_json(filename: str) -> BaseballElimination:
    """
    Load data from JSON file.
    
    JSON format:
    {
        "teams": ["Yankees", "Red Sox", ...],
        "wins": [92, 88, ...],
        "losses": [58, 62, ...],
        "games": [[0, 3, ...], [3, 0, ...], ...]
    }
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    remaining = [sum(row) for row in data['games']]
    return BaseballElimination(
        data['teams'], data['wins'], data['losses'], 
        remaining, data['games']
    )


def load_interactive() -> BaseballElimination:
    """Interactive command-line input."""
    print("\n=== Baseball Elimination Data Entry ===\n")
    
    n = int(input("Number of teams: "))
    
    teams, wins, losses = [], [], []
    print("\nEnter team data:")
    for i in range(n):
        name = input(f"  Team {i+1} name: ")
        w = int(input(f"  {name} wins: "))
        l = int(input(f"  {name} losses: "))
        teams.append(name)
        wins.append(w)
        losses.append(l)
    
    print("\nEnter remaining games between teams:")
    games = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = int(input(f"  {teams[i]} vs {teams[j]}: "))
            games[i][j] = games[j][i] = g
    
    remaining = [sum(row) for row in games]
    return BaseballElimination(teams, wins, losses, remaining, games)


def display_results(be: BaseballElimination):
    """Display elimination results."""
    print("\n" + "=" * 50)
    print("ELIMINATION ANALYSIS")
    print("=" * 50)
    
    # Standings
    print(f"\n{'Team':<15} {'W':<5} {'L':<5} {'Rem':<5} {'Max W':<6}")
    print("-" * 40)
    for i, team in enumerate(be.teams):
        max_w = be.wins[i] + be.remaining[i]
        print(f"{team:<15} {be.wins[i]:<5} {be.losses[i]:<5} "
              f"{be.remaining[i]:<5} {max_w:<6}")
    
    # Elimination status
    print(f"\n{'Team':<15} {'Status':<12} {'Certificate':<25}")
    print("-" * 55)
    for i, team in enumerate(be.teams):
        is_elim, cert = be.is_eliminated(i)
        status = "ELIMINATED" if is_elim else "ALIVE"
        cert_str = ", ".join(cert) if cert else "-"
        print(f"{team:<15} {status:<12} {cert_str:<25}")


def main():
    """Main entry point with menu."""
    print("\n=== Baseball Elimination Algorithm ===")
    print("\nHow would you like to load data?")
    print("  1. Enter manually")
    print("  2. Load from CSV files")
    print("  3. Load from JSON file")
    print("  4. Use example data")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == "1":
        be = load_interactive()
    elif choice == "2":
        standings = input("Standings CSV file: ").strip()
        games = input("Games matrix CSV file: ").strip()
        be = load_from_csv(standings, games)
    elif choice == "3":
        filename = input("JSON file: ").strip()
        be = load_from_json(filename)
    else:
        # Example data
        teams = ["Atlanta", "Philadelphia", "New York", "Montreal"]
        wins = [83, 80, 78, 77]
        losses = [71, 79, 78, 82]
        games = [[0,1,6,1], [1,0,0,2], [6,0,0,0], [1,2,0,0]]
        remaining = [sum(row) for row in games]
        be = BaseballElimination(teams, wins, losses, remaining, games)
    
    display_results(be)
    
    # Optional visualization
    try:
        from visualization import visualize_elimination_network
        show_viz = input("\nShow network visualization? (y/n): ").strip().lower()
        if show_viz == 'y':
            idx = int(input("Team index to visualize (0-based): "))
            visualize_elimination_network(be, idx, show_flow=True)
    except ImportError:
        pass


if __name__ == "__main__":
    main()