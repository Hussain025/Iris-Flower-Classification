"""
Iris Flower Classification
Covers: data exploration, visualization, train/test split,
        training three classifiers, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ── 1. Load & Explore ────────────────────────────────────────────────────────

df = pd.read_csv("Iris.csv")
df.drop(columns=["Id"], inplace=True)          # Id is not a feature

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nClass distribution:\n{df['Species'].value_counts()}")
print(f"\nDescriptive statistics:\n{df.describe()}")

# ── 2. Visualisations ────────────────────────────────────────────────────────

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
species_palette = {
    "Iris-setosa": "#4C72B0",
    "Iris-versicolor": "#DD8452",
    "Iris-virginica": "#55A868",
}

# 2a. Histograms per feature
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle("Feature Distributions by Species", fontsize=14, fontweight="bold")
for ax, feat in zip(axes.flat, features):
    for species, color in species_palette.items():
        subset = df[df["Species"] == species][feat]
        ax.hist(subset, bins=15, alpha=0.6, label=species, color=color)
    ax.set_title(feat)
    ax.set_xlabel("cm")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("histograms.png", dpi=120)
plt.close()
print("\n[Saved] histograms.png")

# 2b. Pairplot (scatter matrix)
pair_fig = sns.pairplot(df, hue="Species", palette=species_palette,
                        diag_kind="kde", plot_kws={"alpha": 0.6})
pair_fig.fig.suptitle("Pairplot of Iris Features", y=1.02, fontsize=13, fontweight="bold")
pair_fig.savefig("pairplot.png", dpi=120)
plt.close()
print("[Saved] pairplot.png")

# 2c. Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
            linewidths=0.5, square=True)
ax.set_title("Feature Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.close()
print("[Saved] correlation_heatmap.png")

# ── 3. Prepare Data ──────────────────────────────────────────────────────────

X = df[features].values
le = LabelEncoder()
y = le.fit_transform(df["Species"])          # 0=setosa, 1=versicolor, 2=virginica

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 4. Train Classifiers ─────────────────────────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {"model": model, "y_pred": y_pred,
                     "accuracy": accuracy_score(y_test, y_pred)}

# ── 5. Evaluate ──────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

class_names = le.classes_

for name, res in results.items():
    print(f"\n── {name} ──")
    print(f"  Accuracy : {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")
    print(classification_report(y_test, res["y_pred"],
                                target_names=class_names, zero_division=0))

# 5a. Confusion matrices (side-by-side)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAcc={res['accuracy']:.2%}", fontsize=10)
    ax.tick_params(axis="x", labelrotation=15)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=120)
plt.close()
print("\n[Saved] confusion_matrices.png")

# 5b. Accuracy comparison bar chart
fig, ax = plt.subplots(figsize=(7, 4))
names = list(results.keys())
accs  = [results[n]["accuracy"] for n in names]
bars  = ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"], width=0.5)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison", fontweight="bold")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{acc:.2%}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=120)
plt.close()
print("[Saved] accuracy_comparison.png")

# 5c. Decision Tree visualisation
dt_model = results["Decision Tree"]["model"]
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(dt_model, feature_names=features, class_names=class_names,
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Decision Tree Structure (max_depth=4)", fontweight="bold")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=120)
plt.close()
print("[Saved] decision_tree.png")

# ── 6. Summary ───────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
best = max(results, key=lambda n: results[n]["accuracy"])
for name, res in results.items():
    marker = " ◄ best" if name == best else ""
    print(f"  {name:<25} {res['accuracy']:.2%}{marker}")
print("\nAll plots saved as PNG files in the working directory.")
