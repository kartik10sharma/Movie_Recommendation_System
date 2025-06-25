import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for MultiLabelBinarizer
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        
    def fit(self, X, y=None):
        return self.mlb.fit(X)
    
    def transform(self, X):
        return self.mlb.transform(X)

# Load dataset
df = pd.read_csv("netflix_list.csv")

# Clean numerical columns
for col in ["runtime", "numVotes", "rating"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN in critical columns
df.dropna(subset=["runtime", "numVotes", "rating"], inplace=True)

# Safely parse 'cast' column
df["cast"] = df["cast"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
)

# Filter top 20 actors
top_actors = pd.Series([actor for sublist in df["cast"] for actor in sublist])\
                .value_counts().nlargest(20).index
df["cast"] = df["cast"].apply(lambda cast: [actor for actor in cast if actor in top_actors])

# Features and target
X = df[["type", "runtime", "cast", "language", "isAdult", "numVotes"]]
y = df["rating"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
    ("language", OneHotEncoder(handle_unknown="ignore"), ["language"]),
    ("cast", MultiLabelBinarizerTransformer(), "cast")
], remainder="passthrough")

# Full pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "movie_recommender_rf_model.pkl")
