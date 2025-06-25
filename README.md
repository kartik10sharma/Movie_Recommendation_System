# 🍿 Movie Rating Prediction App (Netflix Dataset)

A Flask-based machine learning web application that predicts a movie or TV show's rating using metadata like type, runtime, language, number of votes, and cast.  
This app is powered by a Random Forest Regressor trained on Netflix's movie & TV show metadata.

---

## 📊 Dataset

**Source:** [Kaggle - Netflix TV Shows and Movie List](https://www.kaggle.com/datasets/snehaanbhawal/netflix-tv-shows-and-movie-list)

**Description:**  
This dataset contains detailed metadata for thousands of Netflix titles, including:

- `type` (Movie or TV Show)
- `title`
- `director`
- `cast` (list of actors)
- `country`
- `date_added`
- `release_year`
- `rating`
- `duration` (used as runtime)
- `listed_in`
- `description`
- `language`
- `numVotes` *(synthetically generated or inferred)*

---

## 🧠 Machine Learning Model

- **Algorithm:** Random Forest Regressor
- **Features Used**:
  - `type` (One-Hot Encoded)
  - `runtime` (numeric)
  - `language` (One-Hot Encoded)
  - `numVotes` (numeric)
  - `isAdult` (binary)
  - `cast` (MultiLabelBinarized for top 20 frequent actors)
- **Target**: `rating` (IMDb-style rating)
- **Model Output**: A predicted float value representing the expected rating.

---

## 💻 Web App Features

- Simple form UI to enter movie details.
- Backend processing using a trained `movie_recommender_rf_model.pkl`.
- Real-time prediction of IMDb-style rating.
- Outputs the rating on the result page.

---

## 📁 Project Structure

```

movie-rating-app/
├── movie\_model\_trainer.py          # Script to train and save the model
├── movie\_recommender\_rf\_model.pkl # Saved trained model
├── app.py                          # Main Flask application
├── templates/
│   ├── index.html                  # Input form for movie details
│   └── result.html                 # Page displaying predicted rating
└── static/                         # (optional) for styling/CSS

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/movie-rating-app.git
cd movie-rating-app
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

```python
# Run this script if you want to retrain the model
python movie_model_trainer.py
```

### 4. Run the Flask App

```bash
python app.py
```

### 5. Open in Browser

Go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧾 Sample Input Fields

* Type: `Movie` or `TV Show`
* Runtime: `e.g., 105` (minutes)
* Cast: `Leonardo DiCaprio, Tom Hanks` (comma-separated, must be among top 20)
* Language: `English`
* Is Adult: `0` for No, `1` for Yes
* Votes: `e.g., 250000`

---

## ✅ Requirements

```
Flask
pandas
numpy
scikit-learn
joblib
```

Install via:

```bash
pip install Flask pandas numpy scikit-learn joblib
```

---

## 🙋‍♂️ Author

**Kartik Sharma**
*Artificial Intelligence & Machine Learning Student*
[GitHub](https://github.com/kartik10sharma) | [LinkedIn](https://linkedin.com)

---

## 📄 License

This project is for educational purposes only.
Dataset © [Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/netflix-tv-shows-and-movie-list)

---



