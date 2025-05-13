import pandas as pd
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt



csv_path = "/Users/joslinishimwe/Documents/Spring2025/computationalLingustics/FinalProject/BaseSheet_CLEANED.csv"
csv_df = pd.read_csv(csv_path)

linear_a_words = [
    word.strip()
    for word in csv_df["NEW FORMAT"].dropna()
    if "-" in word and "[" not in word and "]" not in word and "+" not in word
]
syllable_sequences = [word.split('-') for word in linear_a_words]


transition_counts = defaultdict(lambda: defaultdict(int))
total_counts = defaultdict(int)

for syllables in syllable_sequences:
    for i in range(len(syllables) - 1):
        first = syllables[i]
        second = syllables[i + 1]
        transition_counts[first][second] += 1
        total_counts[first] += 1

transition_probs = defaultdict(dict)
for first in transition_counts:
    for second in transition_counts[first]:
        transition_probs[first][second] = transition_counts[first][second] / total_counts[first]

flat_transitions = [
    (first, second, count)
    for first in transition_counts
    for second, count in transition_counts[first].items()
]

sorted_transitions = sorted(flat_transitions, key=lambda x: x[2], reverse=True)

top_k = 10  # You can adjust this value to get more or fewer transitions
top_transitions = sorted_transitions[:top_k]

print("Top", top_k, "Most Frequent Transitions:")
for transition in top_transitions:
    print(f"Transition: {transition[0]} -> {transition[1]}, Count: {transition[2]}")


matrix_df = pd.DataFrame(transition_probs).fillna(0)

annotation_threshold = 0.05
annotations = matrix_df.applymap(lambda x: f"{x:.2f}" if x > annotation_threshold else "")

# Plot the heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(
    matrix_df,
    cmap="Blues",
    annot=annotations,
    fmt="s",
    linewidths=.5,
    cbar_kws={'label': 'Transition Probability'}
)

plt.title("Linear A Syllable Transition Matrix (Probabilities)")
plt.xlabel("Next Syllable")
plt.ylabel("Current Syllable")
plt.tight_layout()
plt.show()
