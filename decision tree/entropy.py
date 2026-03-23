import numpy as np

# ===========================
# 1. ENTROPY
# ===========================
def calculate_entropy(y):
    """
    Calculates the entropy of a list/array of labels.
    H(S) = - sum(p_i * log2(p_i))
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # only consider non-zero probabilities (log2(0) is undefined)
    entropy_val = -np.sum(probabilities * np.log2(probabilities))
    return entropy_val


# ===========================
# 2. INFORMATION GAIN
# ===========================
def information_gain(data, feature_index, target_index):
    """
    Computes information gain of splitting 'data' on the given feature.
    IG(S, A) = H(S) - sum( |Sv|/|S| * H(Sv) )
    """
    parent_entropy = calculate_entropy(data[:, target_index])

    # unique values the feature can take
    feature_values = np.unique(data[:, feature_index])

    # weighted entropy of children after splitting
    weighted_child_entropy = 0
    for value in feature_values:
        subset = data[data[:, feature_index] == value]   # rows where feature == value
        weight = len(subset) / len(data)                 # |Sv| / |S|
        weighted_child_entropy += weight * calculate_entropy(subset[:, target_index])

    return parent_entropy - weighted_child_entropy


# ===========================
# 3. BEST SPLIT
# ===========================
def best_split(data, target_index, used_features):
    """
    Finds the feature column with the highest information gain.
    Skips the target column and any already-used features.
    Returns the index of the best feature, or None if no gain.
    """
    num_features = data.shape[1]
    best_gain = -1
    best_feature = None

    for col in range(num_features):
        if col == target_index or col in used_features:
            continue
        gain = information_gain(data, col, target_index)
        if gain > best_gain:
            best_gain = gain
            best_feature = col

    return best_feature


# ===========================
# 4. BUILD TREE (recursive)
# ===========================
def build_tree(data, target_index, feature_names, used_features=None, depth=0, max_depth=10):
    """
    Recursively builds a decision tree using the ID3 algorithm.
    Returns a nested dictionary representing the tree.
    """
    if used_features is None:
        used_features = set()

    labels = data[:, target_index]

    # --- Base case 1: all labels are the same (pure node) ---
    if len(np.unique(labels)) == 1:
        return labels[0]                    # leaf node

    # --- Base case 2: no more features to split on, or max depth ---
    if len(used_features) == data.shape[1] - 1 or depth >= max_depth:
        # return majority label
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]    # leaf node

    # --- Find best feature ---
    feature_col = best_split(data, target_index, used_features)
    if feature_col is None:
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]

    feature_name = feature_names[feature_col]
    tree = {feature_name: {}}

    # mark this feature as used
    new_used = used_features | {feature_col}

    # --- Split data on each unique value of the best feature ---
    for value in np.unique(data[:, feature_col]):
        subset = data[data[:, feature_col] == value]
        if len(subset) == 0:
            # no data for this branch — use parent majority
            values, counts = np.unique(labels, return_counts=True)
            tree[feature_name][value] = values[np.argmax(counts)]
        else:
            tree[feature_name][value] = build_tree(
                subset, target_index, feature_names, new_used, depth + 1, max_depth
            )

    return tree


# ===========================
# 5. PRINT TREE
# ===========================
def print_tree(tree, indent=""):
    """
    Pretty-prints the decision tree with indentation.
    """
    if not isinstance(tree, dict):
        print(f"{indent}└── Predict: {tree}")
        return

    for feature_name, branches in tree.items():
        print(f"{indent}[{feature_name}]")
        for value, subtree in branches.items():
            print(f"{indent}  ├── {feature_name} == {value}")
            print_tree(subtree, indent + "  │   ")


# ===========================
# 6. PREDICT
# ===========================
def predict(tree, sample, feature_names):
    """
    Predicts the label for a single sample by walking the tree.
    """
    if not isinstance(tree, dict):
        return tree  # reached a leaf

    feature_name = list(tree.keys())[0]
    feature_index = feature_names.index(feature_name)
    feature_value = sample[feature_index]

    # look up which branch to follow
    subtree = tree[feature_name].get(feature_value, None)
    if subtree is None:
        return "Unknown (value not seen in training)"

    return predict(subtree, sample, feature_names)


# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    # --- Input ---
    rows = int(input("Enter number of rows: "))
    cols_input = input("Enter feature names separated by space (last = target), or press Enter to auto-generate: ").strip()

    data = []
    for _ in range(rows):
        row = list(map(int, input("Enter row (space-separated integers): ").split()))
        data.append(row)

    data = np.array(data)
    num_cols = data.shape[1]
    target_index = num_cols - 1   # last column is always the target

    # feature names
    if cols_input:
        feature_names = cols_input.split()
    else:
        feature_names = [f"F{i}" for i in range(num_cols - 1)] + ["Target"]

    print(f"\n--- Dataset ({rows} rows, {num_cols} cols) ---")
    print(f"Features: {feature_names[:-1]}")
    print(f"Target  : {feature_names[-1]}")
    print(data)

    # --- Build ---
    print("\n--- Building Decision Tree ---")
    tree = build_tree(data, target_index, feature_names)

    # --- Print ---
    print("\n--- Decision Tree ---")
    print_tree(tree)

    # --- Predict ---
    print("\n--- Prediction ---")
    try_more = "y"
    while try_more.lower() == "y":
        sample = list(map(int, input(f"Enter {num_cols - 1} feature values (space-separated): ").split()))
        result = predict(tree, sample, feature_names)
        print(f"Prediction: {result}")
        try_more = input("Try another? (y/n): ").strip()

    print("Done!")