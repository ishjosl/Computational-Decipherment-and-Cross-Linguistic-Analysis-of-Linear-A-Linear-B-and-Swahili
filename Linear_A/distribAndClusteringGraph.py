import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict

csv_df = pd.read_csv("/Users/joslinishimwe/Documents/Spring2025/computationalLingustics/FinalProject/BaseSheet_Cleaned.csv")

valid_new_words = [
    word.strip()
    for word in csv_df["NEW FORMAT"].dropna()
    if "-" in word and "[" not in word and "]" not in word and "+" not in word
]

syllable_sequences = [word.split('-') for word in valid_new_words]

transition_counts = defaultdict(lambda: defaultdict(int))

for syllables in syllable_sequences:
    for i in range(len(syllables) - 1):
        first = syllables[i]
        second = syllables[i + 1]
        transition_counts[first][second] += 1

G = nx.DiGraph()
for from_syll, to_dict in transition_counts.items():
    for to_syll, count in to_dict.items():
        G.add_edge(from_syll, to_syll, weight=count)

clustering_dict = nx.clustering(G.to_undirected())
clustering_values = list(clustering_dict.values())

plt.figure(figsize=(8, 6))
sns.histplot(clustering_values, bins=10, kde=True, color="teal")
plt.title("Clustering Coefficient Distribution (Linear A Syllable Graph)")
plt.xlabel("Clustering Coefficient")
plt.ylabel("Number of Syllables")
plt.tight_layout()
plt.savefig("LinearA_Clustering_Distribution.jpeg", format="jpeg", dpi=600)
plt.show()

top_k = 25
top_edges = sorted(
    [(u, v, d['weight']) for u, v, d in G.edges(data=True)],
    key=lambda x: x[2], reverse=True
)[:top_k]

G_top = nx.DiGraph()
G_top.add_weighted_edges_from(top_edges)

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G_top, seed=42)
nx.draw(G_top, pos, with_labels=True, node_size=1200, node_color="skyblue", edge_color="gray", font_size=10)
nx.draw_networkx_edge_labels(G_top, pos, edge_labels={(u, v): w for u, v, w in top_edges}, font_size=8)

plt.title("Top-25 Syllable Transitions in Linear A")
plt.tight_layout()
plt.savefig("LinearA_Top25_Transition_Graph.jpeg", format="jpeg", dpi=600)
plt.show()
