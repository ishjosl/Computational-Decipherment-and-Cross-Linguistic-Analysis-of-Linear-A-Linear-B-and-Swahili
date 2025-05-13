import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1. Load the Linear B Word List
# -----------------------------
input_file = '/Users/princengiruwonsanga/Downloads/915_final/linear_b_filtered_long_words.txt'
output_pairs_csv = '/Users/princengiruwonsanga/Downloads/915_final/linear_b_trans_data.csv'

with open(input_file, 'r', encoding='utf-8') as file:
    content = file.read()

# Split and clean the transcriptions
linear_b_words = [word.strip('"') for word in content.split(", ") if word.strip()]

# -----------------------------
# 2. Preprocess Words into Syllables
# -----------------------------
syllable_sequences = [word.split('-') for word in linear_b_words]

# -----------------------------
# 3. Build Transition Counts
# -----------------------------
transition_counts = defaultdict(lambda: defaultdict(int))
total_counts = defaultdict(int)

for syllables in syllable_sequences:
    for i in range(len(syllables) - 1):
        first = syllables[i]
        second = syllables[i + 1]
        transition_counts[first][second] += 1
        total_counts[first] += 1

# -----------------------------
# 4. Normalize into Transition Probabilities
# -----------------------------
transition_probs = defaultdict(dict)

for first in transition_counts:
    for second in transition_counts[first]:
        prob = transition_counts[first][second] / total_counts[first]
        transition_probs[first][second] = prob

# -----------------------------
# 5. Visualize the Markov Model
# -----------------------------
matrix_df = pd.DataFrame(transition_probs).fillna(0)

annotation_threshold = 0.05
annotations = matrix_df.apply(lambda row: row.map(lambda x: f"{x:.2f}" if x > annotation_threshold else ""), axis=1)

plt.figure(figsize=(12, 7))
sns.heatmap(matrix_df, cmap="Blues", annot=annotations, fmt="s", linewidths=0.5, cbar_kws={'label': 'Transition Probability'})
plt.title("Linear B Syllable Transition Matrix (High-Confidence Words)", fontsize=16)
plt.xlabel("Next Syllable")
plt.ylabel("Current Syllable")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Export CSV of Syllable Pairs + Probability
# -----------------------------
# Ensure the output directory exists
os.makedirs(os.path.dirname(output_pairs_csv), exist_ok=True)

rows = []
for first in transition_probs:
    for second in transition_probs[first]:
        transition = f"{first}-{second}"
        probability = round(transition_probs[first][second], 4)
        rows.append({"Syllable_Pair": transition, "Probability": probability})

df_pairs = pd.DataFrame(rows)
df_pairs.to_csv(output_pairs_csv, index=False)
print(f"\n✅ Exported {len(df_pairs)} transition pairs to: {output_pairs_csv}")
