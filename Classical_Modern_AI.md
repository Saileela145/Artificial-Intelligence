
## 🧠 Part 1: Classical AI – The Evolution of Search Algorithms

---

### 1️⃣ Breadth-First Search (BFS)
**Where It Started & Why:**  
Early AI researchers needed a way to **guarantee the shortest path** in an unweighted graph. BFS explores the graph level-by-level, ensuring you find the shortest path if one exists.

**Concept:**  
- Start from the root node.  
- Visit all neighbors first, then their neighbors.  
- Uses a **Queue** (FIFO).

**Drawbacks:**  
- Very high memory usage on large graphs.  
- Slow if the goal node is deep in the search tree.

---

### 2️⃣ Depth-First Search (DFS)
**Why It Came Next:**  
BFS was memory-hungry. DFS was introduced to **reduce memory requirements** by going deep before backtracking.

**Concept:**  
- Start from the root.  
- Follow a branch as far as possible before backtracking.  
- Uses a **Stack** (LIFO) or recursion.

**Drawbacks:**  
- Does not guarantee the shortest path.  
- Can get stuck in infinite loops in cyclic graphs without checks.

---

### 3️⃣ Best First Search
**Why It Came Next:**  
BFS and DFS explored blindly. Best First Search used a **heuristic** to guide exploration toward the goal faster.

**Concept:**  
- Always expand the most promising node (lowest heuristic value `h(n)`).  
- Uses a **Priority Queue**.

**Drawbacks:**  
- Not guaranteed to find the optimal path.  
- Poor heuristics can lead to very bad performance.

---

### 4️⃣ A* Search
**Why It Came Next:**  
Best First Search didn’t guarantee shortest paths. A* combines **cost so far (`g`)** with **estimated cost to goal (`h`)** to find the optimal path.

**Formula:**  
`f(n) = g(n) + h(n)`

**Drawbacks:**  
- Memory intensive for large graphs.  
- Needs a well-designed (admissible) heuristic.

---

### 5️⃣ AO* Search
**Why It Came Next:**  
A* works for OR graphs (choose one path). AO* handles **AND-OR graphs**, where multiple subproblems may need solving together.

**Concept:**  
- Expands the most promising node while considering AND (all children must be solved) and OR (one child suffices).  
- Useful for planning and hierarchical tasks.

**Drawbacks:**  
- More complex to implement.  
- Requires structured problem definitions.

---
### ​​​6️⃣ Iterative Deepening DFS (IDDFS)
**Why It Came Next:**  
DFS has poor path length awareness; BFS uses too much memory. IDDFS combines their strengths.

**Concept:**  
- Perform DFS to a depth limit, then incrementally increase the limit.  
- Ensures optimality like BFS, with memory efficiency like DFS.

**Drawbacks:**  
- Repeated depth-limited searches increase computation cost.

---

### ​​​7️⃣ Iterative Deepening A\* (IDA\*)
**Why It Came Next:**  
A\* is optimal but memory-heavy. IDA\* aims to use less memory.

**Concept:**  
- Conduct depth-first searches using cost thresholds (f-value limits), increasing thresholds iteratively.  
- Achieves A\*-like optimality with linear memory cost.

**Drawbacks:**  
- Can significantly expand nodes multiple times, increasing computation.

---

### 8️⃣​​​ Hill Climbing (Greedy Ascent)
**Why It Came Next:**  
Need for simple, local search when full path planning isn't viable.

**Concept:**  
- Start from an initial state, repeatedly move to neighboring states that improve the heuristic.  
- Stops when no neighbor is better.

**Drawbacks:**  
- Gets stuck in local maxima.  
- No guarantee of reaching the global optimum.

---

### ​​​9️⃣ Beam Search
**Why It Came Next:**  
Full breadth search (BFS) is expensive. Beam search limits breadth.

**Concept:**  
- At each level, keep only the top **K** best nodes (beam width).  
- Prunes the rest to manage memory/time.

**Drawbacks:**  
- Risk of discarding paths that lead to optimal solutions.  
- Quality depends on beam width.

---


## 🤖 Part 2: Modern AI – Machine Learning Story (detailed)

---

### 1️⃣ Data Preprocessing
**Why It Came First:**  
Raw datasets often have missing values, inconsistent formats, or irrelevant data. Models perform poorly without clean input.

**Common Steps:**  
- **Handle missing values:** remove rows, or impute with mean/median/mode.  
- **Encode categorical features:**  
  - *Label encoding* (category → integer)  
  - *One-hot encoding* (create binary columns)  
- **Remove duplicates** and irrelevant columns.  
- **Feature engineering:** create new useful features from raw ones.

**Caution:**  
- Bad preprocessing can introduce bias. Always inspect data distributions.

---

### 2️⃣ Feature Scaling
**Why It Came Next:**  
Features with different magnitudes can dominate distance-based or gradient-based algorithms.

**Main Methods:**  
- **Standardization (Z-score):**  
  `z = (x - mean) / std` — result has mean ≈ 0 and std ≈ 1.  
- **Min-Max Scaling:**  
  `x' = (x - min) / (max - min)` — result scaled to `[0, 1]`.

**When to use:**  
- Use scaling for KNN, SVM, logistic regression (often), neural networks.  
- Tree-based models (decision trees, random forests) usually don’t require scaling.

---

### 3️⃣ Train–Test Split
**Why It’s Important:**  
Testing a model on the same data it trained on gives overly optimistic results.

**Typical Practice:**  
- Split dataset into **training set** and **testing set** (common ratio: `80% train / 20% test`).  
- Optionally use **validation set** or **cross-validation** (e.g., k-fold CV) for hyperparameter tuning.

**Goal:**  
- Ensure the model generalizes to unseen data.

---

### 4️⃣ Model Evaluation Metrics
**Why More Than Accuracy:**  
Accuracy is misleading for imbalanced datasets. Use precision/recall/F1 to understand behavior.

**Key Metrics:**  
- **Accuracy:** `(TP + TN) / (TP + TN + FP + FN)`  
- **Precision:** `TP / (TP + FP)` — of predicted positives, how many are correct  
- **Recall (Sensitivity):** `TP / (TP + FN)` — of actual positives, how many were found  
- **F1-score:** `2 * (precision * recall) / (precision + recall)` — balance of precision & recall

**Confusion Matrix:**  
|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | TP                 | FN                 |
| Actual Negative | FP                 | TN                 |

**Use cases:**  
- For fraud detection (rare positive class) prefer high recall.  
- For spam detection prefer high precision to avoid false positive mistakes.

---

### 5️⃣ Overfitting & Underfitting
**Definitions:**  
- **Overfitting:** Model learns noise & detail in training data → high training accuracy, poor test accuracy.  
- **Underfitting:** Model too simple → poor accuracy on both training and test.

**How to detect:**  
- Compare training vs test performance (or use cross-validation).

**How to fix Overfitting:**  
- Get more training data.  
- Use regularization (L1/L2).  
- Use dropout (for NN), pruning (for trees).  
- Use simpler model or reduce features.  
- Use ensemble methods (bagging).

**How to fix Underfitting:**  
- Increase model complexity (deeper tree, more features, a neural network).  
- Improve features (feature engineering).  
- Decrease regularization.

---

### 6️⃣ Logistic Regression
**Why It Appears Early:**  
Simple, interpretable model for binary classification and a good baseline.

**Concept:**  
- Model predicts probability `P(y=1 | x)` using the logistic (sigmoid) function on a linear combination of features.  
- Decision threshold (commonly 0.5) maps probability → class.

**Drawbacks:**  
- Assumes a linear decision boundary.  
- Can underperform on complex, non-linear data.

---

### 7️⃣ Decision Trees
**Why They Came Next:**  
To model non-linear relationships and create human-readable decision rules.

**Concept:**  
- Split data by choosing features and thresholds that maximize "purity" of child nodes.  
- Splitting criteria: **Gini impurity** or **Entropy (information gain)**.  
  - `Gini = 1 - sum(p_i^2)`  
  - `Entropy = -sum(p_i * log2(p_i))`

**Drawbacks:**  
- Prone to overfitting.  
- Unstable: small changes in data can produce different trees.

---

### 8️⃣ Random Forests
**Why They Came Next:**  
To reduce variance and overfitting of single decision trees.

**Concept:**  
- Build many trees on random subsets of data and random subsets of features (bagging + feature randomness).  
- For classification: use majority vote of trees.  
- For regression: average tree outputs.

**Drawbacks:**  
- Less interpretable than one tree.  
- More computationally heavy.

