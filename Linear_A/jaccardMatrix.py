import pandas as pd
from collections import Counter

linearA = pd.read_csv("/Users/joslinishimwe/Documents/Spring2025/computationalLingustics/FinalProject/linearAtransitionsList.csv")  # Columns: ["Syllable_Pair", ...]
linearB = pd.read_csv("/Users/joslinishimwe/Documents/Spring2025/computationalLingustics/FinalProject/linear_b_trans_data.csv")
swahili = pd.read_csv("/Users/joslinishimwe/Documents/Spring2025/computationalLingustics/FinalProject/transitionsSwahili.csv")

print("Linear A columns:", linearA.columns.tolist())
print("Linear B columns:", linearB.columns.tolist()) 
print("Swahili columns:", swahili.columns.tolist())

def get_syllable_pairs(df, column_name=None):
    if column_name:
        return set(df[column_name].dropna())
    else:
        for col in ['Syllable_Pair', 'Pair', 'bigram', 'transition']:
            if col in df.columns:
                return set(df[col].dropna())
        return set(df.iloc[:, 0].dropna())

set_linearA = get_syllable_pairs(linearA)
set_linearB = get_syllable_pairs(linearB) 
set_swahili = get_syllable_pairs(swahili)

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

jac_linearA_linearB = jaccard_similarity(set_linearA, set_linearB)
jac_linearA_swahili = jaccard_similarity(set_linearA, set_swahili) 
jac_linearB_swahili = jaccard_similarity(set_linearB, set_swahili)

print("\nJaccard Similarity Results:")
print(f"Linear A ↔ Linear B: {jac_linearA_linearB:.4f}")
print(f"Linear A ↔ Swahili: {jac_linearA_swahili:.4f}")
print(f"Linear B ↔ Swahili: {jac_linearB_swahili:.4f}")

similarity_matrix = pd.DataFrame({
    'Linear A': [1.0, jac_linearA_linearB, jac_linearA_swahili],
    'Linear B': [jac_linearA_linearB, 1.0, jac_linearB_swahili],
    'Swahili': [jac_linearA_swahili, jac_linearB_swahili, 1.0]
}, index=['Linear A', 'Linear B', 'Swahili'])

print("\nSimilarity Matrix:")
print(similarity_matrix)

overlap_AB = set_linearA & set_linearB
print(f"\nShared bigrams between Linear A and Linear B ({len(overlap_AB)}):")
print(overlap_AB)

print("\nSimilarity Matrix:")
print(similarity_matrix)

overlap_AS = set_linearA & set_swahili
print(f"\nShared bigrams between Linear A and swahili ({len(overlap_AS)}):")
print(overlap_AS)

overlap_BS = set_linearB & set_swahili
print(f"\nShared bigrams between Linear B and swahili ({len(overlap_BS)}):")
print(overlap_BS)


linearA_syllables = Counter([syl for pair in set_linearA for syl in pair.split('-')])
linearB_syllables = Counter([syl for pair in set_linearB for syl in pair.split('-')])
swahili_syllables = Counter([syl for pair in set_linearB for syl in pair.split('-')])


shared_syllables = set(linearA_syllables) & set(linearB_syllables) & set(swahili_syllables)
print(f"\nShared syllables: {shared_syllables}")


try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Jaccard Similarity Between Scripts")
    plt.tight_layout()
    plt.savefig("jaccard_similarity_heatmap.png")
    print("\nSaved visualization to jaccard_similarity_heatmap.png")
except ImportError:
    print("\nVisualization libraries not available - install seaborn and matplotlib for heatmap")