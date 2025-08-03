# 📖 Logistic Regression Theory

## 1️⃣ What is Logistic Regression?
- A **supervised learning algorithm** used for **classification** (mainly **binary classification**: Yes/No, 0/1, Pass/Fail).
- Despite its name, it is **not** for regression — it predicts **categories**.
- Output is a **probability** between **0 and 1**.

## 2️⃣ Why Logistic Regression Instead of Linear Regression?
- **Linear regression** can give values outside `0–1`, which are not valid probabilities.
- **Logistic regression** fixes this using a **Sigmoid function**.

## 3️⃣ Sigmoid Function
**Formula:**
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

- Squashes any number into the range **0 to 1**.
- Looks like an **S-shaped curve**.

**Example:**
- Input `z = 0` → Output = 0.50
- Input `z = 3` → Output ≈ 0.95
- Input `z = -3` → Output ≈ 0.05

## 4️⃣ Decision Rule (Threshold)
- **If** Probability ≥ 0.5 → **Class = 1** (Positive)
- **If** Probability < 0.5 → **Class = 0** (Negative)

**Example:**
- Probability = `0.87` → **Pass**
- Probability = `0.23` → **Fail**

## 5️⃣ Training Logistic Regression
- **Input:** Features (`X`) and labels (`y`).
- The model finds the **best weights** and **bias** that fit the data.
- Uses **Maximum Likelihood Estimation (MLE)** to choose parameters that make the observed data most probable.

## 6️⃣ Accuracy
**Formula:**
\[
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

- If the model predicted correctly **80 out of 100** times:
\[
\text{Accuracy} = \frac{80}{100} = 0.8 \ (80\%)
\]

## 7️⃣ Advantages
- Simple and easy to implement.
- Works well for small datasets.
- Gives probabilities, not just classifications.

## 8️⃣ Limitations
- Works best for **linearly separable** data.
- Not suitable for **very complex patterns** without extra preprocessing or feature engineering.

## 9️⃣ Real-World Uses
- Spam email detection (**Spam / Not Spam**)
- Student pass/fail prediction
- Loan approval prediction
- Medical diagnosis (**Has disease / No disease**)

  # EXAMPLE :

 # 🎯 Logistic Regression – Predict Student Pass/Fail

We want to predict whether a student will **pass** or **fail** based on how many hours they study.

## 📂 Step 1: Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

| Library / Function                                  | What it is                                              | Why we use it                              | When to use                                                     | If not imported                                         |
| --------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------- | ------------------------------------------------------- |
| **pandas**                                          | Python library for working with tables (rows & columns) | Easy to store, clean, and manipulate data  | Whenever you work with structured data (CSV, Excel, DataFrames) | No `pd.DataFrame`, must handle raw lists/dicts manually |
| **train\_test\_split** *(sklearn.model\_selection)* | Function to split data                                  | Split into **training** & **testing** sets | Almost every ML project to avoid overfitting                    | Manual splitting (slow, error‑prone)                    |
| **LogisticRegression** *(sklearn.linear\_model)*    | Logistic Regression model                               | Classify into categories (0/1, Yes/No)     | Binary classification problems                                  | Can't create model object → **NameError**               |
| **accuracy\_score** *(sklearn.metrics)*             | Function to measure correct prediction ratio            | Check how accurate model is                | After predictions to evaluate performance                       | Must write your own formula
         |
## Step 2:Create Dataset

data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'pass_exam':   [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

|Column         |Meaning                |Example              |Why use DataFrame                  | If not used              |
|---------------|-----------------------|---------------------|-----------------------------------|--------------------------|
|**study_hours**|Hours studied          |`5` → studied 5 hours|Easy table handling                |Must manage lists manually|
|**pass_exam**  |Result(1=pass,0 = fail)|`1` → pass           |Pandas gives filter,sort,statistics|Complex manual handling   |

 ## Step 3:Separate Features & Labels

X = df[['study_hours']]  # Features
y = df['pass_exam']      # Labels

| Variable | Meaning         | Why needed                             | If not done                     |
| -------- | --------------- | -------------------------------------- | ------------------------------- |
| **X**    | Input variables | Model needs inputs to make predictions | Model won’t know input          |
| **y**    | Correct answers | Model learns target output             | Model won’t know correct output |

## Step 4:Split into Training & Testing Data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)
| Parameter           | Meaning               | Why important                      | If not split                         |
| ------------------- | --------------------- | ---------------------------------- | ------------------------------------ |
| **test\_size=0.3**  | 30% test, 70% train   | Balance between training & testing | Fake high accuracy (**overfitting**) |
| **random\_state=1** | Same split every time | Reproducibility                    | Different results each run           |

## Step 5:Create & Train the Model

model = LogisticRegression()
model.fit(X_train, y_train)

| Function                 | Meaning              | If not done        |
| ------------------------ | -------------------- | ------------------ |
| **LogisticRegression()** | Create model object  | Can't fit data     |
| **fit()**                | Teach model patterns | **NotFittedError** |

## Step 6:Make Predictions

y_pred = model.predict(X_test)

| Function      | Meaning                    | If not done             |
| ------------- | -------------------------- | ----------------------- |
| **predict()** | Guess results for new data | Can't evaluate accuracy |

## Step 7:Check Accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

| Function              | Meaning                                 | If not done                   |
| --------------------- | --------------------------------------- | ----------------------------- |
| **accuracy\_score()** | Compare predictions with actual results | No idea how good the model is |

## Step 8:Predict for New Student

hours = [[4.5]]  # 4.5 hours studied
predicted_class = model.predict(hours)
predicted_prob = model.predict_proba(hours)

print(f"Predicted Class: {predicted_class}")
print(f"Probability of Passing: {predicted_prob[0][1]:.2f}")

| Function             | Meaning                         | If not done            |
| -------------------- | ------------------------------- | ---------------------- |
| **predict\_proba()** | Show probability for each class | Only get Yes/No answer |

### Example Output:

Accuracy: 1.0
Predicted Class: [1]
Probability of Passing: 0.87

Accuracy = 100% (on test data)
Predicted class = Pass
Probability = 87% confidence

## 📌 Remember :

pandas → Data handling (tables)
scikit-learn (sklearn) → Machine Learning toolkit
model_selection → Data splitting tools
linear_model → Regression models (Logistic Regression here)
metrics → Model evaluation tools
Logistic Regression → For classification problems, outputs probabilities
Always split data into train/test sets → Avoid overfitting
Accuracy = correct predictions ÷ total predictions
Probability = 87% confidence
