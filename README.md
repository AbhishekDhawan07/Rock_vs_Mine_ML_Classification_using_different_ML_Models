<div align="center">

# 🪨 Rock vs Mine - Multi-Model ML Classification

### *Sonar Signals. Four Algorithms. One Winner.* 🏆

<br>

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SVM](https://img.shields.io/badge/SVM-Classifier-8E44AD?style=for-the-badge&logo=python&logoColor=white)
![Decision Tree](https://img.shields.io/badge/Decision_Tree-Classifier-E74C3C?style=for-the-badge&logo=python&logoColor=white)
![Logistic Regression](https://img.shields.io/badge/Logistic_Regression-Classifier-27AE60?style=for-the-badge&logo=python&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random_Forest-🏆_Winner-F39C12?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

<br>

> 🌊 *"Not all signals look the same underwater. ML knows the difference between a rock and a mine - and tells you which model does it best."*

<br>

---

### 🏆 Model Accuracy Showdown

| 🤖 Model | 🏋️ Training Accuracy | 🧪 Testing Accuracy | 📊 Gap | 🔍 Verdict |
|:---:|:---:|:---:|:---:|:---:|
| 🔵 SVM | **86.67%** | **76.19%** | ~10% | ⚠️ Mild Overfitting |
| 🔴 Decision Tree | **100%** *(pruned: 79%)* | **69.05%** | ~38% | ❌ High Overfitting |
| 🟢 Logistic Regression | **84.84%** | **76.19%** | ~8% | 🔄 Slightly Underfitting |
| 🟡 Random Forest | **100%** | **78.57%** | ~21% | 🏆 **Best Overall!** |

---

</div>

<br>

## 📚 Table of Contents

| # | 📌 Section | 🔍 What's Inside |
|:---:|:---|:---|
| 1 | [📁 Repository Structure](#-repository-structure) | Folder tree with all file names & image locations |
| 2 | [🎯 What Is This Project?](#-what-is-this-project) | Problem statement, goal & classification task |
| 3 | [✨ Features](#-features) | All key capabilities of this multi-model project |
| 4 | [📊 Dataset Deep Dive](#-dataset-deep-dive--sonar-signals) | Sonar dataset structure, columns & class balance |
| 5 | [🔬 End-to-End ML Pipeline](#-end-to-end-ml-pipeline) | Visual flow from raw sonar data to final verdict |
| 6 | [🧪 Step-by-Step Notebook Breakdown](#-step-by-step-notebook-breakdown) | All 8 stages with code & insights |
| 7 | [🤖 Model Deep Dives](#-model-deep-dives--four-algorithms-one-arena) | Individual analysis of all 4 trained models |
| 8 | [📉 Overfitting Analysis & Fixes](#-overfitting-analysis--fixes) | How pruning & regularization were applied |
| 9 | [📊 Accuracy Comparison Charts](#-accuracy-comparison-charts) | Training & Testing bar charts from the project |
| 10 | [🏆 Final Verdict](#-final-verdict--which-model-wins) | Ranked conclusions from the project |
| 11 | [🆚 Model Comparison Table](#-model-comparison-table) | Head-to-head across all key metrics |
| 12 | [🚀 Getting Started](#-getting-started) | Clone, install & launch instructions |
| 13 | [📋 Requirements](#-requirements) | All dependencies listed |
| 14 | [📂 File Reference](#-file-reference) | What every file in the repo contains |
| 15 | [🌍 Real-World Applications](#-real-world-applications) | Where sonar classification is used in the real world |
| 16 | [📌 Key Takeaways](#-key-takeaways) | Project highlights & lessons learned |
| 17 | [📜 License](#-license) | MIT License |

---

## 📁 Repository Structure

```
🗂️ Rock_vs_Mine_ML_Classification_using_different_ML_Models/
│
├── 📄 README.md                                                  <- at root level
├── 🖼️ Training Accuracy Comparision of Different Models.png     <- 📊 Bar chart (Training)
├── 🖼️ Testing Accuracy Comparision of Different Models.png      <- 📊 Bar chart (Testing)
│
└── 📂 Different ML Models Project - Rock VS Mine Classification/
    │
    ├── 📓 Different_ML_Models_Project_-_Rock_VS_Mine_Classification.ipynb
    └── 📊 Copy_of_sonar_data.csv       ← SONAR Dataset (208 samples × 61 columns)
```

> 📌 **Note:** The two accuracy comparison images live at the **root of the repository** (outside the project folder), not inside the subfolder.

---

## 🎯 What Is This Project?

This project is a **head-to-head battle of four Machine Learning algorithms** - all trained on real-world sonar signal data - to answer one critical question:

```
🤔  "Is this underwater object a ROCK 🪨 or a MINE 💣?"
```

Using the famous **SONAR Dataset**, this project goes beyond building just one model. It **trains, evaluates, compares, and explains** four completely different classifiers to determine which generalizes best on unseen sonar data.

> 💡 This is a **Binary Classification** task: the model outputs either `R` (Rock) or `M` (Mine) for every sonar signal input.

---

## ✨ Features

```
🪨  BINARY CLASSIFICATION
    └── Classifies sonar signals as Rock (R) or Mine (M)

🤖  FOUR ML MODELS TRAINED & COMPARED
    ├── 🔵 Support Vector Machine (SVM) — Linear Kernel
    ├── 🔴 Decision Tree — with Cost Complexity Pruning (CCP)
    ├── 🟢 Logistic Regression — max_iter=1000
    └── 🟡 Random Forest — 100 estimators + regularization

📊  RICH VISUALIZATIONS
    ├── Feature value distribution (Histogram + KDE)
    ├── Feature correlation heatmap (coolwarm)
    ├── Pairplot of first 10 features
    ├── Boxplot for outlier detection
    └── Training & Testing accuracy bar charts (color-coded)

🩺  OVERFITTING DIAGNOSIS PER MODEL
    └── Every model analyzed for train-test gap and labeled accordingly

🔧  OVERFITTING FIXES APPLIED
    ├── Decision Tree → Cost Complexity Pruning (ccp_alpha=0.05)
    └── Random Forest → max_depth, max_features, max_samples tuning

🏆  WINNER DECLARED WITH EVIDENCE
    └── Random Forest wins with best test accuracy (78.57%) & smallest gap

📓  WELL-DOCUMENTED NOTEBOOK
    └── 42 cells with conclusions written after every single model

♻️  FULLY REPRODUCIBLE
    └── Fixed random_state=42 across all four models
```

---

## 📊 Dataset Deep Dive — SONAR Signals

The **SONAR Dataset** (originally from the UCI Machine Learning Repository) contains sonar signal readings bounced off either a **metal cylinder (mine)** or a **rock** on the ocean floor.

<div align="center">

| 📋 Property | 🔢 Value |
|:---|:---:|
| Total Samples | **208** |
| Total Columns | **61** (60 features + 1 label) |
| Feature Type | Continuous float values (0.0 – 1.0) |
| Target Column | `Column 60` — `R` (Rock) or `M` (Mine) |
| 🪨 Rock Samples | **97** (46.6%) |
| 💣 Mine Samples | **111** (53.4%) |
| Missing Values | ✅ None |
| Duplicates | ✅ None |

</div>

<br>

```
📡 How the Data Works
├── Each row = one sonar signal "ping" fired at an underwater object
├── Columns 0–59 = energy values at 60 different frequency bands
├── Values range from 0.0 (low energy) to 1.0 (high energy)
└── Column 60 = human-labeled answer: R (Rock) or M (Mine)
```

> 🔬 **Why 60 features?** Each sonar "ping" is analyzed across 60 different frequency sweeps. The pattern of energy across these frequencies reveals whether the object is metallic (mine) or natural (rock).

---

## 🔬 End-to-End ML Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  📥  LOAD DATA             →   Read Copy_of_sonar_data.csv           │
│         ↓                                                            │
│  🔍  EDA & VISUALIZATION   →   Shape, Stats, Heatmap, Boxplot        │
│         ↓                                                            │
│  🏷️  FEATURE / TARGET SEP  →   X = cols 0–59  |  y = col 60         │
│         ↓                                                            │
│  🔀  TRAIN-TEST SPLIT      →   80% Train | 20% Test (random=42)      │
│         ↓                                                            │
│  🔵  TRAIN SVM             →   Linear kernel SVC                     │
│         ↓                                                            │
│  🔴  TRAIN DECISION TREE   →   Default → Pruned (ccp_alpha=0.05)     │
│         ↓                                                            │
│  🟢  TRAIN LOG. REGRESSION →   max_iter=1000                         │
│         ↓                                                            │
│  🟡  TRAIN RANDOM FOREST   →   100 trees → Regularized version       │
│         ↓                                                            │
│  📊  VISUALIZE RESULTS     →   Color-coded accuracy bar charts       │
│         ↓                                                            │
│  🏆  FINAL CONCLUSIONS     →   Rank all 4 models, crown winner       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Step-by-Step Notebook Breakdown

<details>
<summary><b>📦 Step 1 - Importing Libraries</b> 🖱️ click to expand</summary>
<br>

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

> 🔧 Four classifiers imported from `sklearn` - all trained and evaluated in the same pipeline for a fair, apples-to-apples comparison.

</details>

---

<details>
<summary><b>📥 Step 2 - Loading & Exploring the Dataset</b> 🖱️ click to expand</summary>
<br>

```python
sonar_data = pd.read_csv("Copy_of_sonar_data.csv")

sonar_data.shape        # → (208, 61)
sonar_data.describe()   # → Stats for all 60 features
sonar_data.info()       # → All float64 + 1 object column
sonar_data.iloc[:, 60].value_counts()  # → M: 111 | R: 97
```

| 🔎 Check | 📋 Finding |
|:---|:---|
| Dataset Shape | `(208, 61)` |
| Missing Values | ✅ Zero |
| Duplicate Rows | ✅ Zero |
| Class Balance | 🟡 Slightly imbalanced - 111 Mines vs 97 Rocks |
| Feature Range | 0.0 - 1.0 (energy values) |

</details>

---

<details>
<summary><b>📊 Step 3 - Visualization & EDA</b> 🖱️ click to expand</summary>
<br>

Four powerful visualizations were generated to understand the data:

| 📈 Plot | 💡 What It Reveals |
|:---|:---|
| 📶 Histogram + KDE | Overall distribution of all 60 feature values |
| 🌡️ Correlation Heatmap | Which frequency bands correlate with each other |
| 🔵 Pairplot (first 10 features) | Feature-to-feature relationships & cluster separation |
| 📦 Boxplot of all features | Outlier presence across each frequency band |

```python
# Feature distribution
sns.histplot(sonar_data.iloc[:, :-1].values.flatten(), bins=50, kde=True, color='blue')

# Correlation heatmap
sns.heatmap(sonar_data.iloc[:, :-1].corr(), cmap='coolwarm', linewidths=0.5)

# Pairplot (first 10 features only for clarity)
sns.pairplot(sonar_data.iloc[:, :10])

# Boxplot
sns.boxplot(data=sonar_data.iloc[:, :-1])
```

</details>

---

<details>
<summary><b>✂️ Step 4 - Feature/Target Split & Train-Test Split</b> 🖱️ click to expand</summary>
<br>

```python
# Features: all 60 sonar frequency columns
X = sonar_data.iloc[:, :-1]   # shape: (208, 60)

# Target: Rock or Mine
y = sonar_data.iloc[:, -1]    # shape: (208,)  → 'R' or 'M'

# 80/20 split, fixed seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

```
📊 Split Summary
├── 🏋️  Training Set   ->  166 samples  (80%)
└── 🧪  Testing Set    ->   42 samples  (20%)
```

</details>

---

<details>
<summary><b>🤖 Step 5–8 - Training All Four Models</b> 🖱️ click to expand</summary>
<br>

Each model was trained, evaluated on both train and test sets, and analyzed for overfitting:

```python
# SVM
svm_model = SVC(kernel='linear', random_state=42)

# Decision Tree (with CCP pruning)
dt_model = DecisionTreeClassifier(ccp_alpha=0.05, random_state=42)

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
```

Each model follows the same evaluation pattern:
```python
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
```

</details>

---

## 🤖 Model Deep Dives - Four Algorithms, One Arena

<details>
<summary><b>🔵 Model 1 - Support Vector Machine (SVM)</b> 🖱️ click to expand</summary>
<br>

```python
svm_model = SVC(kernel='linear', random_state=42)
```

| 📊 Metric | 🔢 Value |
|:---|:---:|
| 🏋️ Training Accuracy | **86.67%** |
| 🧪 Testing Accuracy | **76.19%** |
| 📉 Gap | ~10% |

**💬 Verdict: ⚠️ Mild Overfitting (Acceptable)**
- SVM learned good patterns but slightly struggles to generalize
- A ~10% gap is not alarming — the model is still useful in practice
- Linear kernel may not fully capture the nonlinear nature of sonar data

</details>

---

<details>
<summary><b>🔴 Model 2 — Decision Tree</b> 🖱️ click to expand</summary>
<br>

**🚨 Before Pruning:**

| 📊 Metric | 🔢 Value |
|:---|:---:|
| 🏋️ Training Accuracy | **100%** |
| 🧪 Testing Accuracy | **61.90%** |
| 📉 Gap | **~38%** ← 🚨 High Overfitting |

**✅ After Cost Complexity Pruning (`ccp_alpha=0.05`):**

```python
dt_model = DecisionTreeClassifier(ccp_alpha=0.05, random_state=42)
```

| 📊 Metric | 🔢 Value |
|:---|:---:|
| 🏋️ Training Accuracy | **~79%** |
| 🧪 Testing Accuracy | **~69%** |
| 📉 Gap | Significantly reduced ✅ |

**💬 Verdict: ❌ → ⚠️ High Overfitting fixed to Manageable**
- Default Decision Tree memorized the training data completely
- CCP Pruning trimmed unnecessary branches - trading training perfection for real-world generalization

</details>

---

<details>
<summary><b>🟢 Model 3 - Logistic Regression</b> 🖱️ click to expand</summary>
<br>

```python
lr_model = LogisticRegression(random_state=42, max_iter=1000)
```

| 📊 Metric | 🔢 Value |
|:---|:---:|
| 🏋️ Training Accuracy | **84.84%** |
| 🧪 Testing Accuracy | **76.19%** |
| 📉 Gap | ~8% |

**💬 Verdict: 🔄 Slightly Underfitting (But Stable)**
- Training and testing accuracy are very close — excellent generalization
- The model is stable and trustworthy, not memorizing data
- Slight underfitting because sonar signals have complex nonlinear patterns that logistic regression (a linear model) can't fully capture

</details>

---

<details>
<summary><b>🟡 Model 4 — Random Forest 🏆</b> 🖱️ click to expand</summary>
<br>

**Default Configuration:**

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
```

| 📊 Metric | 🔢 Value |
|:---|:---:|
| 🏋️ Training Accuracy | **100%** |
| 🧪 Testing Accuracy | **78.57%** ← 🏆 Highest! |
| 📉 Gap | ~21% |

**After Extreme Regularization:**

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=1,       # Each tree makes only ONE decision
    max_features=3,    # Each tree sees only 3 random features
    max_samples=0.4,   # Each tree trains on only 40% of data
    random_state=42
)
```

**💬 Verdict: 🏆 High Overfitting but Best Overall Performance**
- Even with overfitting, Random Forest's ensemble averaging keeps test accuracy the highest
- 100 trees working together reduces variance dramatically vs a single Decision Tree
- Best generalization despite perfect training accuracy

</details>

---

## 📉 Overfitting Analysis & Fixes

```
┌─────────────────────────────────────────────────────────────────────┐
│                   🩺 OVERFITTING DIAGNOSIS                          │
├──────────────────┬──────────────┬──────────────┬────────────────────┤
│   🤖 Model        │  🏋️ Train    │  🧪 Test     │  🔍 Status         │
├──────────────────┼──────────────┼──────────────┼────────────────────┤
│  🔵 SVM           │   86.67%    │   76.19%     │  ⚠️  Mild          │
│  🔴 Decision Tree │   100%      │   61.90%     │  🚨  Severe        │
│  🟢 Logistic Reg  │   84.84%    │   76.19%     │  🔄  Underfitting  │
│  🟡 Random Forest │   100%      │   78.57%     │  ⚠️  Moderate      │
└──────────────────┴──────────────┴──────────────┴────────────────────┘

🔧 FIXES APPLIED
├── 🔴 Decision Tree  →  Cost Complexity Pruning (ccp_alpha=0.05)
│                         Gap: 38% → significantly reduced ✅
└── 🟡 Random Forest  →  max_depth=1, max_features=3, max_samples=0.4
                          Ensemble averaging naturally reduces variance ✅
```

---

## 📊 Accuracy Comparison Charts

The following bar charts were generated directly from the notebook, comparing all four models side-by-side. Both images are stored at the **root of the repository**.

<br>

### 🏋️ Training Accuracy Comparison

![Training Accuracy Comparison](Training%20Accuracy%20Comparision%20of%20Different%20Models.png)

<br>

### 🧪 Testing Accuracy Comparison

![Testing Accuracy Comparison](Testing%20Accuracy%20Comparision%20of%20Different%20Models.png)
<br>

> 📌 **Reading the Charts:**
> - 🔵 **Blue** = SVM &nbsp;&nbsp; 🔴 **Red** = Decision Tree &nbsp;&nbsp; 🟢 **Green** = Logistic Regression &nbsp;&nbsp; 🟡 **Orange** = Random Forest
> - In the **training** chart, Decision Tree and Random Forest both hit 100% — a red flag for overfitting
> - In the **testing** chart, Random Forest (orange) edges out all others — confirming it as the winner

---

## 🏆 Final Verdict — Which Model Wins?

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  🥇  BEST OVERALL MODEL      →  🟡 Random Forest                   │
│       Reason: Highest testing accuracy (78.57%)                    │
│                                                                     │
│  🥈  BEST BALANCED MODEL     →  🟡 Random Forest                   │
│       Reason: Smallest train-test gap among high performers         │
│                                                                     │
│  🥉  MOST STABLE MODEL       →  🟢 Logistic Regression             │
│       Reason: Consistent, no memorization, reliable baseline        │
│                                                                     │
│  ⚠️   MOST OVERFITTING        →  🔴 Decision Tree (before pruning)  │
│       Reason: 100% train vs 61.9% test — 38% gap                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🆚 Model Comparison Table

| 📊 Metric | 🔵 SVM | 🔴 Decision Tree | 🟢 Logistic Reg | 🟡 Random Forest |
|:---|:---:|:---:|:---:|:---:|
| 🏋️ Training Accuracy | 86.67% | 79%* | 84.84% | 100% |
| 🧪 Testing Accuracy | 76.19% | 69.05% | 76.19% | **78.57%** 🏆 |
| 📉 Train-Test Gap | ~10% | ~10%* | ~8% | ~21% |
| 🔄 Overfitting Risk | ⚠️ Mild | ⚠️ Managed | 🔄 Underfit | ⚠️ Moderate |
| 🔧 Fix Applied | ❌ None | ✅ CCP Pruning | ❌ None | ✅ Regularized |
| ⏱️ Training Speed | Fast | Fast | Fast | Moderate |
| 🌳 Interpretability | ⚠️ Moderate | ✅ High | ✅ High | ❌ Low |
| 🎯 Best For | Balanced data | Explainability | Linear patterns | Best accuracy |

*After Cost Complexity Pruning (ccp_alpha=0.05)*

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Rock_vs_Mine_ML_Classification_using_different_ML_Models.git
cd "Rock_vs_Mine_ML_Classification_using_different_ML_Models/Different ML Models Project - Rock VS Mine Classification"
```

### 2️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3️⃣ Launch the Notebook
```bash
jupyter notebook "Different_ML_Models_Project_-_Rock_VS_Mine_Classification.ipynb"
```

> ⚠️ **Quick Fix Required:** Update the dataset path in Cell 4 from the hardcoded Windows path to:
> ```python
> sonar_data = pd.read_csv("Copy_of_sonar_data.csv")
> ```

---

## 📋 Requirements

```
Python         >= 3.7
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 📂 File Reference

| 📄 File | 📂 Location | 📋 What's Inside |
|:---|:---:|:---|
| `Different_ML_Models_Project_-_Rock_VS_Mine_Classification.ipynb` | 📂 Project Folder | Full pipeline: EDA → 4 Models → Charts → Conclusions (42 cells) |
| `Copy_of_sonar_data.csv` | 📂 Project Folder | 208 sonar signal samples × 60 features + 1 label (R/M) |
| `Training Accuracy Comparision of Different Models.png` | 🗂️ Repo Root | Bar chart comparing training accuracy across all 4 models |
| `Testing Accuracy Comparision of Different Models.png` | 🗂️ Repo Root | Bar chart comparing testing accuracy across all 4 models |

---

## 🌍 Real-World Applications

<div align="center">

> 🚢 **Naval Defense** — autonomous underwater mine detection to protect vessels
>
> 🤿 **Ocean Exploration** — distinguishing natural rock formations from debris or man-made objects
>
> 🛳️ **Autonomous Submarines** — real-time sonar classification for navigation & threat avoidance
>
> 🏗️ **Seabed Mapping** — identifying object types on the ocean floor for geological surveys
>
> 🔬 **Signal Processing Research** — benchmark dataset for testing new ML classification algorithms

</div>

---

## 📌 Key Takeaways

```
✅  Four ML models trained & compared on the same dataset — fair & rigorous
✅  208 sonar signal samples, 60 frequency features — real-world complexity
✅  Overfitting diagnosed for every single model with evidence
✅  Decision Tree overfitting fixed using Cost Complexity Pruning (CCP)
✅  Random Forest regularized using max_depth, max_features, max_samples
✅  Random Forest wins — highest test accuracy (78.57%) despite overfitting
✅  Logistic Regression is the most stable and trustworthy linear baseline
✅  Color-coded bar charts make accuracy comparisons immediately visual
✅  All models use random_state=42 — 100% reproducible results
✅  Final conclusions written after every model — clear, structured insights
```

---

## 📜 License

Distributed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

### 💬 *"The best model isn't always the most complex — it's the one that generalizes best to the real world."*

<br>

⭐ **Found this useful? Drop a star and share the knowledge!** ⭐

`🪨 Built with passion for ML + Multi-Model Benchmarking`

</div>
