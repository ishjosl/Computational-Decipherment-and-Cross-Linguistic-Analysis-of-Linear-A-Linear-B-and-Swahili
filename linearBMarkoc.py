import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# 1. Load Linear B Words from File
# -----------------------------
# Read from the file
input_file = '/Users/princengiruwonsanga/Downloads/915_final/formatted_transcriptions.txt'  # <-- path to your formatted transcriptions

with open(input_file, 'r', encoding='utf-8') as file:
    content = file.read()

# Split the big line into a list
linear_b_words = [word.strip('"') for word in content.split(", ") if word.strip()]

# -----------------------------
# 2. Preprocess Words into Syllable Sequences
# -----------------------------
syllable_sequences = []
for word in linear_b_words:
    syllables = word.split('-')
    syllable_sequences.append(syllables)

# -----------------------------
# (the rest of your original code stays the same!)
# -----------------------------
# Build Transition Counts
transition_counts = defaultdict(lambda: defaultdict(int))

for syllables in syllable_sequences:
    for i in range(len(syllables) - 1):
        first = syllables[i]
        second = syllables[i + 1]
        transition_counts[first][second] += 1

# Create Raw Transition Matrix
unique_syllables = sorted({syll for word in syllable_sequences for syll in word})
count_matrix = pd.DataFrame(0, index=unique_syllables, columns=unique_syllables)

for first in unique_syllables:
    for second in unique_syllables:
        count_matrix.loc[first, second] = transition_counts.get(first, {}).get(second, 0)

# Visualize Raw Counts
plt.figure(figsize=(20, 18))
sns.heatmap(
    count_matrix,
    cmap="YlGnBu",
    annot=True,
    fmt="d",
    linewidths=0.5,
    cbar_kws={"label": "Transition Count"}
)

plt.title("Linear B Syllable Transition Matrix (Raw Counts)", fontsize=16)
plt.xlabel("Next Syllable", fontsize=14)
plt.ylabel("Current Syllable", fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("Linear_b_Syllable_Transition_Matrix_Raw_Counts.jpeg")  # Save figure
plt.show()
