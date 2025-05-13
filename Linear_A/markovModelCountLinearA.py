import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

csv_df = pd.read_csv("/Users/joslinishimwe/Documents/Spring2025/computationalLingustics/FinalProject/BaseSheet_Cleaned.csv")
new_csv_words = csv_df["NEW FORMAT"].dropna()
new_csv_words = [word.strip() for word in new_csv_words if "-" in word]

linear_a_words = []

all_linear_a_words = linear_a_words + new_csv_words

valid_new_words = [
    word.strip()
    for word in csv_df["NEW FORMAT"].dropna()
    if "-" in word and "[" not in word and "]" not in word and "+" not in word
]

syllable_sequences = [word.split('-') for word in all_linear_a_words]

transition_counts = defaultdict(lambda: defaultdict(int))

for syllables in syllable_sequences:
    for i in range(len(syllables) - 1):
        first = syllables[i]
        second = syllables[i + 1]
        transition_counts[first][second] += 1

unique_syllables = sorted({syll for seq in syllable_sequences for syll in seq})
count_matrix = pd.DataFrame(0, index=unique_syllables, columns=unique_syllables)

for first in unique_syllables:
    for second in unique_syllables:
        count_matrix.loc[first, second] = transition_counts.get(first, {}).get(second, 0)

top_transitions = []

for first, transitions in transition_counts.items():
    for second, count in transitions.items():
        top_transitions.append((first, second, count))

top_transitions = sorted(top_transitions, key=lambda x: x[2], reverse=True)[:10]

print("Top 10 Most Frequent Syllable Transitions:")
for i, (first, second, count) in enumerate(top_transitions, start=1):
    print(f"{i}. {first} → {second}: {count} times")

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
