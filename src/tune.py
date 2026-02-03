import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

df = pd.read_csv("data/processed/full_preprocessed.csv")
X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    "C": [0.1, 1.0, 10.0],
    "solver": ["lbfgs", "liblinear"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=3000),
    param_grid=param_grid,
    scoring="f1",
    cv=5
)
grid.fit(X_train, y_train)

best = grid.best_estimator_
pred = best.predict(X_test)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, zero_division=0)
rec = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

joblib.dump(best, "models/model_tuned.joblib")

text = (
    f"Best Params: {grid.best_params_}\n"
    f"Accuracy:  {acc:.4f}\n"
    f"Precision: {prec:.4f}\n"
    f"Recall:    {rec:.4f}\n"
    f"F1 Score:  {f1:.4f}\n"
)
open("reports/tuning_results.txt", "w", encoding="utf-8").write(text)
print(text)
print("Saved models/model_tuned.joblib and reports/tuning_results.txt")
