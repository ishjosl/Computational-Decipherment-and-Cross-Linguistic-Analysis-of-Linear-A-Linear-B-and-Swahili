import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm
import re

# Load and preprocess the Swahili text
def load_and_preprocess():
    # Get the directory of the current Python script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the file path relative to the script's location
    file_path = os.path.join(current_dir, 'AFYA_Cleaned.txt')

    def split_into_syllables(word):
        if len(word) > 3:  # Split words longer than 3 characters into syllables
            chunks = [word[i:i+3] for i in range(0, len(word), 3)]
            # Ensure no single-character chunks
            if len(chunks) > 1 and len(chunks[-1]) == 1:
                chunks[-2] += chunks[-1]  # Merge the last single character with the previous chunk
                chunks.pop()  # Remove the last chunk
            return chunks
        return [word]  # Return the word as is if it's not too long

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        processed_words = [syllable for word in words for syllable in split_into_syllables(word)]
    return processed_words

# Build a Markov transition matrix with progress bar
def build_markov_model(words):
    transitions = defaultdict(lambda: defaultdict(int))
    for i in tqdm(range(len(words) - 1), desc="Building Markov Model"):
        transitions[words[i]][words[i + 1]] += 1

    # Convert counts to probabilities
    transition_counts = pd.DataFrame(transitions).fillna(0).T
    transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
    return transition_counts, transition_probabilities

# Compute Jaccard similarity matrix
def compute_jaccard_similarity(matrix_df):
    jaccard_matrix = pd.DataFrame(index=matrix_df.index, columns=matrix_df.columns, dtype=float)
    for i in matrix_df.index:
        for j in matrix_df.columns:
            set_i = set(matrix_df.loc[i][matrix_df.loc[i] > 0].index)
            set_j = set(matrix_df.loc[j][matrix_df.loc[j] > 0].index)
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard_matrix.loc[i, j] = intersection / union if union > 0 else 0
    return jaccard_matrix

# Filter matrices to include only top-K most frequent words
def filter_top_k_words(transition_counts, transition_probabilities, words, top_k):
    word_frequencies = Counter(words)
    most_common_words = {word for word, _ in word_frequencies.most_common(top_k)}

    filtered_counts = transition_counts.loc[
        transition_counts.index.isin(most_common_words),
        transition_counts.columns.isin(most_common_words)
    ]
    filtered_probabilities = transition_probabilities.loc[
        transition_probabilities.index.isin(most_common_words),
        transition_probabilities.columns.isin(most_common_words)
    ]
    return filtered_counts, filtered_probabilities

# Plot matrix with professional styling
def plot_matrix(matrix_df, title, cmap=None):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Format values based on whether they're zero, integer, or float
    annot_data = matrix_df.copy().applymap(
        lambda x: "0" if x == 0 else (f"{int(x)}" if float(x).is_integer() else f"{x:.2f}")
    )

    plt.figure(figsize=(20, 16))  # Increase figure size for better clarity
    ax = sns.heatmap(
        matrix_df,
        cmap=cmap,
        cbar=True,
        annot=annot_data,
        fmt="",
        annot_kws={"size": 12},
        linewidths=0,
        linecolor='gray',
        square=True
    )

    # Increase color bar tick font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)  # Adjust font size here

    plt.title(title, fontsize=20, weight='bold')
    plt.xlabel("Next syllable", fontsize=18, weight='bold')
    plt.ylabel("Current syllable", fontsize=18, weight='bold')
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16, rotation=0)
    plt.tight_layout()
    plt.show()




swahili_words = load_and_preprocess()
transition_counts, transition_probabilities = build_markov_model(swahili_words)


def plot_word_frequency(words, top_k=None, offset=0):
    """
    Plots the word frequency for a given range of words.
    :param words: List of words or syllables.
    :param top_k: Number of words to plot.
    :param offset: Starting index for the range of words to plot.
    """
    word_frequencies = Counter(words).most_common(offset + top_k)[offset:offset + top_k]
    words, counts = zip(*word_frequencies)

    plt.figure(figsize=(14, 8))
    sns.barplot(x=list(counts), y=list(words), palette="coolwarm", orient='h')
    plt.title(f"Words {offset + 1} to {offset + top_k} Most Frequent Words/Syllables", fontsize=18, weight='bold')
    plt.xlabel("Frequency", fontsize=18)
    plt.ylabel("Words/Syllables", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

# Plot word frequency for the first top 30 words
plot_word_frequency(swahili_words, top_k=30, offset=0)



# Plot matrices in batches
def plot_matrices_in_batches(words, transition_counts, transition_probabilities, batch_size=30):
    """
    Plots transition counts, probabilities, and Jaccard similarity matrices in batches.
    :param words: List of words or syllables.
    :param transition_counts: Transition counts matrix.
    :param transition_probabilities: Transition probabilities matrix.
    :param batch_size: Number of words to include in each batch.
    """
    word_frequencies = Counter(words).most_common()
    total_words = len(word_frequencies)
    num_batches = (total_words + batch_size - 1) // batch_size  # Calculate the number of batches

    for batch in range(num_batches):
        offset = batch * batch_size
        batch_words = {word for word, _ in word_frequencies[offset:offset + batch_size]}

        # Filter matrices for the current batch
        filtered_counts = transition_counts.loc[
            transition_counts.index.isin(batch_words),
            transition_counts.columns.isin(batch_words)
        ]
        filtered_probabilities = transition_probabilities.loc[
            transition_probabilities.index.isin(batch_words),
            transition_probabilities.columns.isin(batch_words)
        ]

        # Compute Jaccard similarity for the current batch
        jaccard_similarity = compute_jaccard_similarity(filtered_counts)

        # Plot matrices
        plot_matrix(filtered_counts, f"Swahili Syllable Batch {batch + 1}: Transition Counts Matrix", cmap="Blues")
        plot_matrix(filtered_probabilities, f"Swahili Syllable Batch {batch + 1}: Transition Probabilities Matrix", cmap="Greens")
        plot_matrix(jaccard_similarity, f"Swahili Syllable Batch {batch + 1}: Jaccard Similarity Matrix", cmap="Purples")


# Plot matrices in batches
plot_matrices_in_batches(swahili_words, transition_counts, transition_probabilities, batch_size=40)


