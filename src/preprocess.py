import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

RAW_PATH = Path("data/raw/churn.csv")
OUT_DIR = Path("data/processed")
MODEL_DIR = Path("models")

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH)
df.columns = [c.strip() for c in df.columns]

# Try to find the target column
target_candidates = [c for c in df.columns if c.lower() in ("churn", "churned", "label", "target")]
if not target_candidates:
    target_col = df.columns[-1]  # fallback
else:
    target_col = target_candidates[0]

y = df[target_col]
X = df.drop(columns=[target_col])

# Treat object columns as categorical; also treat common integer-coded categoricals as categorical if present
possible_cat = {"tariff plan", "status", "plan", "contract"}
cat_cols = [c for c in X.columns if X[c].dtype == "object" or c.strip().lower() in possible_cat]
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() == 2 else None
)

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop",
)

preprocessor.fit(X_train)

def transform_to_df(X_in: pd.DataFrame, y_in: pd.Series) -> pd.DataFrame:
    X_t = preprocessor.transform(X_in)
    feature_names = preprocessor.get_feature_names_out()
    out = pd.DataFrame(X_t, columns=feature_names)
    out["churn"] = y_in.reset_index(drop=True)
    return out

train_df = transform_to_df(X_train, y_train)
test_df  = transform_to_df(X_test, y_test)
full_df  = transform_to_df(X, y)

train_df.to_csv(OUT_DIR / "train_preprocessed.csv", index=False)
test_df.to_csv(OUT_DIR / "test_preprocessed.csv", index=False)
full_df.to_csv(OUT_DIR / "full_preprocessed.csv", index=False)

joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

print("Saved:")
print(" - data/processed/train_preprocessed.csv")
print(" - data/processed/test_preprocessed.csv")
print(" - data/processed/full_preprocessed.csv")
print(" - models/preprocessor.joblib")
print(f"Target column used: {target_col}")
