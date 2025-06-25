from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

# Custom transformer
class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        return self.mlb.fit(X)

    def transform(self, X):
        return self.mlb.transform(X)

# Initialize app
app = Flask(__name__)

# Load model
model = joblib.load("movie_recommender_rf_model.pkl")

# Define top actors (must match the training phase)
top_actors = [
    "Robert De Niro", "Tom Hanks", "Brad Pitt", "Johnny Depp", "Leonardo DiCaprio",
    "Samuel L. Jackson", "Bruce Willis", "Morgan Freeman", "Matt Damon", "Denzel Washington",
    "Tom Cruise", "Harrison Ford", "Adam Sandler", "Ben Stiller", "Mark Wahlberg",
    "Will Smith", "Nicolas Cage", "Christian Bale", "Liam Neeson", "George Clooney"
]

# Home route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle form submission
@app.route("/predict", methods=["POST"])
def predict():
    data = request.form  # Use request.form to get form data
    input_data = {
        "type": [data["type"]],
        "runtime": [float(data["runtime"])],
        "cast": [[actor.strip() for actor in data["cast"].split(',') if actor.strip() in top_actors]],
        "language": [data["language"]],
        "isAdult": [int(data["adult"])],
        "numVotes": [int(float(data["votes"]))]
    }
    prediction = model.predict(pd.DataFrame(input_data))
    return render_template('result.html', rating=prediction[0])  # Render result.html with the predicted rating

if __name__ == "__main__":
    app.run(debug=True)
