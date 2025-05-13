import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import re
import networkx as nx
from collections import Counter
import pickle
import hashlib
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import csv


#--------------------------------
# Function to compute the Markov model
#--------------------------------
def compute_markov_model(syllables, text, cache_file='markov_model_cache.pkl'):
    """
    Computes the Markov transition counts and probabilities from the given syllables and text.
    Uses caching to avoid re-computation unless parameters change.

    :param syllables: List of syllables from the syllabified data.
    :param text: Input text to analyze.
    :param cache_file: Path to the cache file.
    :return: Tuple of (transition_counts, transition_probabilities).
    """


    # Generate a hash of the input parameters
    param_hash = hashlib.md5((str(syllables) + text).encode('utf-8')).hexdigest()

    # Check if the cache file exists and is valid
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                if cache_data.get('param_hash') == param_hash:
                    print("Loading cached Markov model...")
                    return cache_data['transition_counts'], cache_data['transition_probabilities']
        except (EOFError, pickle.UnpicklingError):
            print("Cache file is corrupted. Recomputing and overwriting the cache...")
            os.remove(cache_file)  # Delete the corrupted cache file

    # Split the text into words and syllables
    words = text.split()
    processed_syllables = []
    for word in tqdm(words, desc="Processing words to extract syllables"):
        for syllable in syllables:
            if syllable in word:
                processed_syllables.append(syllable)

    # Build transition counts
    transitions = defaultdict(lambda: defaultdict(int))
    for i in tqdm(range(len(processed_syllables) - 1), desc="Building Markov Model"):
        transitions[processed_syllables[i]][processed_syllables[i + 1]] += 1

    # Convert counts to DataFrame
    transition_counts = pd.DataFrame(transitions).fillna(0).T

    # Normalize counts to probabilities
    transition_probabilities = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)

    # Save the results to the cache file
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'param_hash': param_hash,
            'transition_counts': transition_counts,
            'transition_probabilities': transition_probabilities
        }, f)

    return transition_counts, transition_probabilities

# Load syllables from the CSV file
# syllables_df = pd.read_csv('syllabified_words.csv')
# syllables = [syllable for word in syllables_df['syllabified'] for syllable in word.split('-')]

# Load portion of syllabified words
syllables_df = pd.read_csv('syllabified_words.csv', nrows=40)  # Read only the first 10 rows
syllables = [syllable for word in syllables_df['syllabified'] for syllable in word.split('-')]


# Read the text from the file
with open('Swahili_Corpus_combined.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()



# Compute Markov model
transition_counts, transition_probabilities = compute_markov_model(syllables, text)



#--------------------------------
# Function to plot clustering coefficient vs. syllable distribution
#--------------------------------
def plot_clustering_coefficient_vs_syllable_distribution(transition_counts):
    """
    Plots the clustering coefficient vs. the density distribution of syllables.

    :param transition_counts: DataFrame representing the transition counts.
    """
    # Create a directed graph from the transition counts
    G = nx.DiGraph()
    for i in transition_counts.index:
        for j in transition_counts.columns:
            weight = transition_counts.loc[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    # Compute clustering coefficients
    clustering_coefficients = nx.clustering(G.to_undirected(), weight='weight')

    # Prepare data for plotting
    coefficients = list(clustering_coefficients.values())

    # Plot density distribution
    plt.figure(figsize=(12, 8))
    sns.kdeplot(coefficients, fill=True, color="blue", alpha=0.6, linewidth=2)
    plt.title("Clustering Coefficient vs. Syllable Distribution", fontsize=20, weight='bold')
    plt.xlabel("Clustering Coefficient", fontsize=16)
    plt.ylabel("Number of Syllables", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save the plot
    plt.savefig('clustering_coefficient_vs_syllable_distribution.png', dpi=300)
    plt.show()

# Example usage
plot_clustering_coefficient_vs_syllable_distribution(transition_counts)




#--------------------------------
# Function to export transition network to CSV
#--------------------------------
def export_transition_network_to_csv(transition_probabilities, output_file='transition_network.csv'):
    """
    Exports the transition network graph to a CSV file with columns 'From', 'To', and 'Probability',
    excluding transitions where the source and target nodes are the same.

    :param transition_probabilities: DataFrame representing the transition probabilities.
    :param output_file: Path to the output CSV file.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['From', 'To', 'Probability'])  # Header row

        for from_node in transition_probabilities.index:
            for to_node in transition_probabilities.columns:
                if from_node != to_node:  # Exclude same-to-same transitions
                    probability = transition_probabilities.loc[from_node, to_node]
                    if probability > 0:  # Include only non-zero probabilities
                        writer.writerow([from_node, to_node, probability])

    print(f"Transition network exported to {output_file}")

export_transition_network_to_csv(transition_probabilities, output_file='transition_network.csv')

def plot_transition_matrix(matrix_df, cmap="Blues"):
    """
    Plots the transition matrix with professional styling.

    :param matrix_df: DataFrame representing the transition matrix.
    :param cmap: Color map for the heatmap.
    """
    plt.figure(figsize=(18, 10))
    ax = sns.heatmap(
        matrix_df,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
        linewidths=0.5,
        linecolor="#d3d3d3",  # Lighter grey color for linecolor
        cbar=True
    )
    # Increase color bar tick font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)  # Adjust font size for color bar ticks
    cbar.set_label("Transition Probability", fontsize=18)  # Add label to color bar

    plt.title("Markov Transition Probability Matrix", fontsize=18, weight="bold")
    plt.xlabel("Next Syllable", fontsize=18)
    plt.ylabel("Current Syllable", fontsize=18)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14, rotation=0)
    plt.tight_layout()
    plt.show()


# Plot transition count matrix
def plot_transition_count_matrix(matrix_df, cmap="Reds"):
    """
    Plots the transition count matrix with professional styling.

    :param matrix_df: DataFrame representing the transition counts.
    :param cmap: Color map for the heatmap.
    """
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(
        matrix_df,
        cmap=cmap,
        annot=False,
        fmt=".0f",
        annot_kws={"size": 10},
        linewidths=0.5,
        linecolor="#d3d3d3",  # Lighter grey color for linecolor
        cbar=True
    )
    # Increase color bar tick font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)  # Adjust font size for color bar ticks
    cbar.set_label("Transition Count", fontsize=18)  # Add label to color bar

    plt.title("Markov Transition Count Matrix", fontsize=18, weight="bold")
    plt.xlabel("Next Syllable", fontsize=18)
    plt.ylabel("Current Syllable", fontsize=18)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14, rotation=0)
    plt.tight_layout()
    plt.show()

# Plot transition count matrix
plot_transition_count_matrix(transition_counts)

# Plot transition probability matrix
plot_transition_matrix(transition_probabilities)
#save the transition probability matrix to a CSV file
transition_probabilities.to_csv('transition_probabilities.csv', index=True, encoding='utf-8')


# Extract transitions in the format A-B
def save_transitions_to_csv(transition_counts, output_file='transitions.csv'):
    """
    Extracts transitions in the format A-B (excluding same-to-same transitions) and saves them to a CSV file.

    :param transition_counts: DataFrame representing the transition counts.
    :param output_file: Path to the output CSV file.
    """
    transitions = []

    for current_syllable in transition_counts.index:
        for next_syllable in transition_counts.columns:
            if current_syllable != next_syllable and transition_counts.loc[current_syllable, next_syllable] > 0:
                transitions.append(f"{current_syllable}-{next_syllable}")

    # Save transitions to a CSV file
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Transition'])  # Header row
        for transition in transitions:
            writer.writerow([transition])

    print(f"Transitions saved to {output_file}")

# Save transitions to a CSV file
save_transitions_to_csv(transition_counts, output_file='transitions.csv')


#--------------------------------
# Function to perform association rule mining with consonants
#--------------------------------
def association_rule_mining_with_consonants(swahili_file, linear_a_file, linear_b_file, min_support=0.2, min_confidence=0.5, max_rows=3):
    def extract_consonants(syllables, script_prefix):
        ini = syllables[0] if len(syllables) > 0 else 'empty'
        med = syllables[1] if len(syllables) > 1 else 'empty'
        fin = syllables[-1] if len(syllables) > 1 else 'empty'
        return f"{script_prefix}_ini_{ini}", f"{script_prefix}_med_{med}", f"{script_prefix}_fin_{fin}"

    print("Loading a portion of Swahili syllables...")
    swahili_data = []
    swahili_df = pd.read_csv(swahili_file, nrows=max_rows)
    for syllables in swahili_df['syllabified'].str.split('-').dropna():
        ini, med, fin = extract_consonants(syllables, "Swahili")
        swahili_data.append({'Script': 'Swahili', 'ini': ini, 'med': med, 'fin': fin})

    print("Loading a portion of Linear A syllables...")
    linear_a_data = []
    linear_a_df = pd.read_csv(linear_a_file, nrows=max_rows)
    for syllables in linear_a_df['NEW FORMAT'].str.split('-').dropna():
        ini, med, fin = extract_consonants(syllables, "LinearA")
        linear_a_data.append({'Script': 'Linear A', 'ini': ini, 'med': med, 'fin': fin})

    print("Loading a portion of Linear B syllables...")
    linear_b_data = []
    linear_b_df = pd.read_csv(linear_b_file, nrows=max_rows)
    for syllables in linear_b_df['Word'].str.split('-').dropna():
        ini, med, fin = extract_consonants(syllables, "LinearB")
        linear_b_data.append({'Script': 'Linear B', 'ini': ini, 'med': med, 'fin': fin})

    print("Combining syllables into a single dataset...")
    combined_data = swahili_data + linear_a_data + linear_b_data
    combined_df = pd.DataFrame(combined_data)

    print("Preparing data for Apriori (one-hot encoding)...")
    transactions = combined_df.groupby('Script').apply(
        lambda x: x[['ini', 'med', 'fin']].values.flatten().tolist()
    ).tolist()
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

    print("Applying Apriori algorithm...")
    frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets.")

    print("Generating association rules...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    print(f"Generated {len(rules)} rules.")

    print("\nAssociation Rules:")
    for _, row in tqdm(rules.iterrows(), desc="Processing rules", total=len(rules)):
        antecedents = [f"{item.split('_')[2]} ({item.split('_')[1]})" for item in row['antecedents']]
        consequents = [f"{item.split('_')[2]} ({item.split('_')[1]})" for item in row['consequents']]
        antecedent_scripts = {item.split('_')[0] for item in row['antecedents']}
        consequent_scripts = {item.split('_')[0] for item in row['consequents']}
        scripts = antecedent_scripts.union(consequent_scripts)  # Combine scripts from both antecedents and consequents
        script_names = ', '.join(scripts)
        print(f"Rule: {script_names} ({', '.join(antecedents)}) => ({', '.join(consequents)})")
        print(f"Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")

    print("\nAssociation Rules:")
    for _, row in tqdm(rules.iterrows(), desc="Processing rules", total=len(rules)):
        antecedents = [f"{item.split('_')[2]} ({item.split('_')[1]})" for item in row['antecedents']]
        consequents = [f"{item.split('_')[2]} ({item.split('_')[1]})" for item in row['consequents']]
        antecedent_scripts = {item.split('_')[0] for item in row['antecedents']}
        consequent_scripts = {item.split('_')[0] for item in row['consequents']}
        scripts = antecedent_scripts.union(consequent_scripts)  # Combine scripts from both antecedents and consequents

        # Ensure Linear B rules are included
        if "LinearB" in scripts:
            script_names = ', '.join(scripts)
            print(f"Rule: {script_names} ({', '.join(antecedents)}) => ({', '.join(consequents)})")
            print(f"Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}")

# Example usage
association_rule_mining_with_consonants('syllabified_words.csv', 'BaseSheet_CLEANED.csv', 'linear_b_data.csv', min_support=0.0000001, min_confidence=0.0000001, max_rows=4)


def plot_word_count_with_labels(folder_path, labels):
    """
    Plots a bar plot showing the number of words in each .txt file in the given folder, labeled by categories.
    The "Social Media" category is plotted on a secondary y-axis.

    :param folder_path: Path to the folder containing .txt files.
    :param labels: Dictionary mapping file names to their respective categories.
    """
    word_counts = {}

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                word_count = len(text.split())  # Count words
                category = labels.get(file_name, "Unknown")  # Get category or default to "Unknown"
                word_counts[category] = word_count  # Use category as the key

    # Separate "Social Media" from other categories
    social_media_count = word_counts.pop("Social Media", None)
    categories = list(word_counts.keys())
    counts = list(word_counts.values())

    # Plot the main bar plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=categories, y=counts, palette="coolwarm", ax=ax1)
    ax1.set_title("Word Count in Each File by Category", fontsize=18, weight='bold')
    ax1.set_xlabel("Category", fontsize=18)
    ax1.set_ylabel("Word Count", fontsize=18)
    ax1.tick_params(axis='x', rotation=45, labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)

    # Plot "Social Media" on a secondary y-axis
    if social_media_count is not None:
        ax2 = ax1.twinx()
        ax2.bar(["Social Media"], [social_media_count], color="teal", label="Social Media")
        ax2.set_ylabel("Social Media Word Count", fontsize=18, color="teal")
        ax2.tick_params(axis='y', labelsize=16, colors="teal")

    plt.tight_layout()
    plt.show()

# Example usage
labels = {
    'AFYA_Cleaned.txt': 'Health',
    'BIASHARA_Cleaned.txt': 'Business',
    'BUNGE_Cleaned.txt': 'Parliament',
    'DINI_Cleaned.txt': 'Religion',
    'ELIMU_Cleaned.txt': 'Education',
    'HABARI_Cleaned.txt': 'News',
    'KILIMO_Cleaned.txt': 'Agriculture',
    'MITANDAO_Cleaned.txt': 'Social Media',
    'MASHIRIKA_Cleaned.txt': 'Non-Governmental Organizations',
    'SERIKALI_Cleaned.txt': 'Government',
    'SHERIA_Cleaned.txt': 'Laws',
    'SIASA_Cleaned.txt': 'Politics'
}

plot_word_count_with_labels('Swahili Corpus', labels)




#--------------------------------
# Function to find the most frequent words in a file
#--------------------------------
def find_most_frequent_words_from_file(file_path, top_k=10):
    """
    Reads a file and finds the most frequent words with progress verbose.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    words = re.findall(r'\b\w+\b', text.lower())

    # Use tqdm to display progress while counting words
    word_counts = Counter()
    for word in tqdm(words, desc="Processing words"):
        word_counts[word] += 1

    return word_counts.most_common(top_k)

top_words = find_most_frequent_words_from_file('Swahili_Corpus_combined.txt', top_k=500)

# Convert to DataFrame for better formatting
df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
print(df)
# Save the DataFrame to a CSV file
df.to_csv('most_frequent_words_Swahili.csv', index=False, encoding='utf-8')




