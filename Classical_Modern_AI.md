art 1 – Classical AI: The Evolution of Search Algorithms
1️⃣ BFS – The First Step
Problem in early AI:
When researchers started working on pathfinding and puzzle-solving in the 1950s–60s, they needed a method that would guarantee the shortest path in unweighted problems.

Solution invented:
Breadth-First Search — Explore the graph level by level, visiting all neighbors before moving deeper.

How it works:

Use a Queue (FIFO) to keep track of frontier nodes.

Visit each node’s neighbors in order.

Strengths:

Finds shortest path in unweighted graphs.

Simple to implement.

Weakness:

High memory usage for wide/deep graphs.

2️⃣ DFS – The Memory Saver
Problem with BFS:
Needed too much memory — storing all frontier nodes was impractical for big graphs.

Solution invented:
Depth-First Search — Dive deep into one path before backtracking.

How it works:

Use a Stack (LIFO) or recursion.

Follow one branch until you can’t go further, then backtrack.

Strengths:

Very low memory usage.

Weakness:

Does not guarantee shortest path.

Can get stuck in deep or infinite loops.

3️⃣ Best First Search – The First “Smart” Search
Problem with DFS/BFS:
Both explored blindly — wasted time on paths that clearly weren’t leading to the goal.

Solution invented:
Best First Search — Use a heuristic function 
ℎ
(
𝑛
)
h(n) to estimate how close a node is to the goal, then always choose the closest-looking node.

Strengths:

Can be much faster if heuristic is good.

Weakness:

Might give wrong (non-optimal) path.

Bad heuristic = bad performance.

4️⃣ A* Search – The Gold Standard
Problem with Best First Search:
Only cared about estimated closeness, not the actual cost taken so far.

Solution invented:
A* — Combines cost so far 
𝑔
(
𝑛
)
g(n) and heuristic 
ℎ
(
𝑛
)
h(n):

𝑓
(
𝑛
)
=
𝑔
(
𝑛
)
+
ℎ
(
𝑛
)
f(n)=g(n)+h(n)
Strengths:

Finds optimal path if heuristic is admissible.

Weakness:

Still memory-heavy for very large graphs.

5️⃣ AO* Search – The Problem Solver for Complex Tasks
Problem with A*:
Couldn’t handle AND-OR problems (tasks requiring multiple sub-tasks).

Solution invented:
AO* — Works on AND-OR graphs, expanding promising nodes and handling dependencies.

Strengths:

Solves multi-step dependent problems.

Weakness:

Complex to implement.

Part 2 – Modern AI: Machine Learning Steps
1️⃣ Data Preprocessing – The Foundation
Problem:
Raw datasets had missing values, mixed formats, and categorical data models couldn’t understand.

Solution:
Preprocessing: clean, encode, and prepare the data.

2️⃣ Feature Scaling – Leveling the Playing Field
Problem:
Features had different ranges — large values dominated smaller ones.

Solution:
Standardization (mean=0, std=1) or Min-Max scaling (0 to 1).

3️⃣ Train-Test Split – Honest Evaluation
Problem:
Testing on the same data used for training gave misleadingly high accuracy.

Solution:
Split into training set and testing set (e.g., 80%/20%).

4️⃣ Evaluation Metrics – Beyond Accuracy
Problem:
Accuracy failed for imbalanced datasets.

Solution:
Precision, recall, F1-score, confusion matrix.

5️⃣ Overfitting/Underfitting – The Balancing Act
Problem:
Models either memorized data (overfit) or missed patterns (underfit).

Solution:
Regularization, cross-validation, better data/features.

