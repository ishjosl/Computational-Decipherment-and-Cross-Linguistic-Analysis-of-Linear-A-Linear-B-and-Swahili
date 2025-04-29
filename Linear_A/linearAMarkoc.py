import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# 1. Load Additional Words from BaseSheet.csv
# -----------------------------
csv_df = pd.read_csv("/Users/joslinishimwe/Documents/Spring2025/computationalLingustics/FinalProject/BaseSheet_Cleaned.csv")

# Filter valid syllabified sequences from the 'NEW FORMAT' column
new_csv_words = csv_df["NEW FORMAT"].dropna()
new_csv_words = [word.strip() for word in new_csv_words if "-" in word]

# -----------------------------
# 2. Original Linear A Word List
# -----------------------------
linear_a_words = []

# -----------------------------
# 3. Combine CSV Words with Original List
# -----------------------------
all_linear_a_words = linear_a_words + new_csv_words
# Only keep rows that contain hyphenated syllables and do not contain brackets or plus signs
valid_new_words = [
    word.strip()
    for word in csv_df["NEW FORMAT"].dropna()
    if "-" in word and "[" not in word and "]" not in word and "+" not in word
]




# -----------------------------
# 4. Preprocess Words into Syllable Sequences
# -----------------------------
syllable_sequences = [word.split('-') for word in all_linear_a_words]

# -----------------------------
# 5. Build Transition Counts
# -----------------------------
transition_counts = defaultdict(lambda: defaultdict(int))

for syllables in syllable_sequences:
    for i in range(len(syllables) - 1):
        first = syllables[i]
        second = syllables[i + 1]
        transition_counts[first][second] += 1

# -----------------------------
# 6. Create Raw Transition Matrix
# -----------------------------
unique_syllables = sorted({syll for seq in syllable_sequences for syll in seq})
count_matrix = pd.DataFrame(0, index=unique_syllables, columns=unique_syllables)

for first in unique_syllables:
    for second in unique_syllables:
        count_matrix.loc[first, second] = transition_counts.get(first, {}).get(second, 0)

# -----------------------------
# 7. Visualize Raw Counts
# -----------------------------
plt.figure(figsize=(20, 18))
sns.heatmap(
    count_matrix,
    cmap="YlGnBu",
    annot=True,
    fmt="d",
    linewidths=0.5,
    cbar_kws={"label": "Transition Count"}
)

plt.title("Expanded Linear A Syllable Transition Matrix (Raw Counts)", fontsize=16)
plt.xlabel("Next Syllable", fontsize=14)
plt.ylabel("Current Syllable", fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("Expanded_Linear_A_Syllable_Transition_Matrix.jpeg", dpi=300)
plt.show()
