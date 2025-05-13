import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict

# -----------------------------
# 1. Load Transcriptions from TXT
# -----------------------------
with open("/Users/princengiruwonsanga/Downloads/915_final/formatted_transcriptions.txt", 'r', encoding='utf-8') as file:
    content = file.read()

# Clean and parse the string list
transcription_list = [word.strip().strip('"') for word in content.split(",") if "-" in word]

# -----------------------------
# 2. Convert to Syllable Sequences
# -----------------------------
syllable_sequences = [word.split('-') for word in transcription_list]

# -----------------------------
# 3. Build Transition Counts
# -----------------------------
transition_counts = defaultdict(lambda: defaultdict(int))

for syllables in syllable_sequences:
    for i in range(len(syllables) - 1):
        first = syllables[i]
        second = syllables[i + 1]
        transition_counts[first][second] += 1

# -----------------------------
# 4. Build Network Graph from Transitions
# -----------------------------
G = nx.DiGraph()
for from_syll, to_dict in transition_counts.items():
    for to_syll, count in to_dict.items():
        G.add_edge(from_syll, to_syll, weight=count)

# -----------------------------
# 5. Clustering Coefficient Distribution
# -----------------------------
clustering_dict = nx.clustering(G.to_undirected())
clustering_values = list(clustering_dict.values())

plt.figure(figsize=(8, 6))
sns.histplot(clustering_values, bins=10, kde=True, color="#0072BD")
plt.title("Clustering Coefficient Distribution (Linear B Syllable Graph)")
plt.xlabel("Clustering Coefficient")
plt.ylabel("Number of Syllables")
plt.tight_layout()
plt.savefig("LinearB_Clustering_Distribution.jpeg", format="jpeg", dpi=600)
plt.show()

# -----------------------------
# 6. Top-k Transition Graph
# -----------------------------
top_k = 25
top_edges = sorted(
    [(u, v, d['weight']) for u, v, d in G.edges(data=True)],
    key=lambda x: x[2], reverse=True
)[:top_k]

# Create subgraph
G_top = nx.DiGraph()
G_top.add_weighted_edges_from(top_edges)

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G_top, seed=42)
nx.draw(G_top, pos, with_labels=True, node_size=1200, node_color="skyblue", edge_color="gray", font_size=10)
nx.draw_networkx_edge_labels(G_top, pos, edge_labels={(u, v): w for u, v, w in top_edges}, font_size=8)

plt.title("Top-25 Syllable Transitions in Linear B")
plt.tight_layout()
plt.savefig("LinearB_Top25_Transition_Graph.jpeg", format="jpeg", dpi=600)
plt.show()
