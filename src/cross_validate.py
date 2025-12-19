import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/processed/full_preprocessed.csv")
X = df.drop(columns=["churn"])
y = df["churn"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=2000)

scores = cross_validate(
    model, X, y, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=False
)

lines = []
for k, v in scores.items():
    if k.startswith("test_"):
        lines.append(f"{k}: mean={v.mean():.4f}, std={v.std():.4f}")

open("reports/cv_results.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
print("Saved reports/cv_results.txt")
