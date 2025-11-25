"""
Network Flow Visualization
Visualizes baseball elimination flow networks using NetworkX.
"""

from typing import Tuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from baseball_elimination import BaseballElimination


def visualize_elimination_network(be: BaseballElimination, team_idx: int, 
                                   figsize: Tuple[int, int] = (14, 10),
                                   show_flow: bool = False) -> None:
    """
    Visualize the flow network for baseball elimination.
    
    Args:
        be: BaseballElimination instance with league data
        team_idx: Index of the team to check for elimination
        figsize: Figure size as (width, height)
        show_flow: If True, solve max flow and show flow values
    """
    G = nx.DiGraph()
    max_wins = be.wins[team_idx] + be.remaining[team_idx]
    team_name = be.teams[team_idx]
    
    edge_labels = {}
    game_nodes = []
    team_nodes = []
    total_games = 0
    
    G.add_node("source", layer=0)
    G.add_node("sink", layer=3)
    
    # Build game and team nodes
    for i in range(be.n):
        if i == team_idx:
            continue
        for j in range(i + 1, be.n):
            if j == team_idx:
                continue
            if be.games[i][j] > 0:
                gnode = f"{be.teams[i][:3]}-{be.teams[j][:3]}"
                game_nodes.append(gnode)
                G.add_node(gnode, layer=1)
                
                G.add_edge("source", gnode, capacity=be.games[i][j])
                edge_labels[("source", gnode)] = str(be.games[i][j])
                total_games += be.games[i][j]
                
                for t_idx, t_name in [(i, be.teams[i]), (j, be.teams[j])]:
                    if t_name not in team_nodes:
                        team_nodes.append(t_name)
                        G.add_node(t_name, layer=2)
                    G.add_edge(gnode, t_name, capacity=float('inf'))
                    edge_labels[(gnode, t_name)] = "∞"
    
    # Team to sink edges
    for i in range(be.n):
        if i == team_idx:
            continue
        t = be.teams[i]
        if t in team_nodes:
            cap = max(0, max_wins - be.wins[i])
            G.add_edge(t, "sink", capacity=cap)
            edge_labels[(t, "sink")] = str(cap)
    
    # Layout positions
    pos = {"source": (0, 0.5), "sink": (3, 0.5)}
    for idx, node in enumerate(game_nodes):
        pos[node] = (1, (idx + 1) / (len(game_nodes) + 1))
    for idx, node in enumerate(team_nodes):
        pos[node] = (2, (idx + 1) / (len(team_nodes) + 1))
    
    # Compute flow if requested
    flow_value, is_eliminated = 0, False
    if show_flow:
        network, _ = be.build_flow_network(team_idx)
        flow_value = network.ford_fulkerson("source", "sink")
        is_eliminated = flow_value < total_games
    
    # Draw graph
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = []
    for node in G.nodes():
        if node == "source": colors.append("#2ecc71")
        elif node == "sink": colors.append("#e74c3c")
        elif node in game_nodes: colors.append("#3498db")
        else: colors.append("#f39c12")
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#7f8c8d", arrows=True,
                          arrowsize=20, connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, 
                                 label_pos=0.3, ax=ax)
    
    title = f"Elimination Network for {team_name}\n"
    title += f"Max possible wins: {max_wins} | Total games: {total_games}"
    if show_flow:
        status = "ELIMINATED" if is_eliminated else "NOT ELIMINATED"
        title += f"\nMax Flow: {flow_value} | {status}"
    
    ax.set_title(title, fontsize=12, fontweight="bold")
    
    legend = [
        mpatches.Patch(color="#2ecc71", label="Source"),
        mpatches.Patch(color="#3498db", label="Game Nodes"),
        mpatches.Patch(color="#f39c12", label="Team Nodes"),
        mpatches.Patch(color="#e74c3c", label="Sink")
    ]
    ax.legend(handles=legend, loc="upper left")
    ax.axis("off")
    plt.tight_layout()
    plt.show()
