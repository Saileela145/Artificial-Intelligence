
## 🧠 Part 1: Classical AI – The Evolution of Search Algorithms

---

### 1️⃣ Breadth-First Search (BFS)
**Why Created:**  
Early AI needed a way to **guarantee the shortest path** in unweighted graphs. BFS explores **level by level**.

**Concept:**  
- Start from the root node.  
- Explore all neighbors first, then their neighbors.  
- Uses a **queue** (FIFO).

**Success:**  
- Guarantees shortest path.  
- Systematic exploration.

**Drawbacks:**  
- High memory usage for large graphs.  
- Slow if the goal is deep.

---

### 2️⃣ Depth-First Search (DFS)
**Why Created:**  
To **reduce memory usage** compared to BFS.

**Concept:**  
- Go deep along one branch, then backtrack.  
- Uses a **stack** (LIFO) or recursion.

**Success:**  
- Low memory usage.  
- Finds a solution without exploring all nodes.

**Drawbacks:**  
- Does not guarantee shortest path.  
- Can get stuck in infinite loops in cyclic graphs.

---

### 3️⃣ Best First Search
**Why Created:**  
To use **heuristics** (estimates about closeness to the goal) instead of blind exploration.

**Concept:**  
- Selects node with lowest heuristic value \( h(n) \).  
- Uses a **priority queue**.

**Success:**  
- Faster if heuristic is good.

**Drawbacks:**  
- Not guaranteed shortest path.  
- Bad heuristic can lead to poor performance.

---

### 4️⃣ A\* Search
**Why Created:**  
To combine **cost so far (g)** and **estimated cost to goal (h)** for better pathfinding.

**Formula:**  
\[
f(n) = g(n) + h(n)
\]

**Success:**  
- Optimal shortest path if heuristic is admissible.  
- Popular in games, GPS navigation.

**Drawbacks:**  
- Memory intensive for large graphs.  
- Needs a good heuristic.

---

### 5️⃣ AO\* Search
**Why Created:**  
For **AND-OR graphs**, where tasks can be dependent or have multiple solution paths.

**Concept:**  
- Expands promising nodes.  
- Handles sub-problems that must be solved together.

**Success:**  
- Solves complex planning problems.

**Drawbacks:**  
- More complex to implement.  
- Needs structured problem definition.

---

## 🤖 Part 2: Modern AI – From Logistic Regression to Trees

---

### 1️⃣ Data Preprocessing
**Why Created:**  
Models failed with messy, inconsistent data. Preprocessing cleans and formats it.

**Steps:**  
- Handle missing values.  
- Encode categorical features.  
- Remove duplicates.

**Drawbacks:**  
- Poor preprocessing can bias results.

---

### 2️⃣ Feature Scaling
**Why Created:**  
Different feature scales caused bias in distance-based models.

**Methods:**  
- **Standardization:** Mean = 0, StdDev = 1.  
- **Min-Max Scaling:** Rescale to [0, 1].

**Drawbacks:**  
- Wrong scaling method can hurt model performance.

---

### 3️⃣ Decision Trees
**Why Created:**  
Linear models (like Logistic Regression) can’t capture complex, non-linear relationships.

**Concept:**  
- Splits data into branches based on feature values.  
- Uses Gini Index or Entropy for splitting.

**Success:**  
- Easy to understand and interpret.  
- Handles numerical and categorical data.

**Drawbacks:**  
- Prone to overfitting.  
- Small changes in data can change the tree.

---

### 4️⃣ Random Forests
**Why Created:**  
To overcome Decision Tree overfitting by averaging multiple trees.

**Concept:**  
- Creates many decision trees on random subsets of data and features.  
- Final output is a vote (classification) or average (regression).

**Success:**  
- More accurate and stable than a single tree.

**Drawbacks:**  
- Less interpretable.  
- Can be slow with large datasets.


