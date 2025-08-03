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
