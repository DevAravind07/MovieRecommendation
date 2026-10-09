# Movie Recommendation

A runnable movie recommendation app using matrix factorization (SVD) from
`scikit-surprise`, with an interactive Streamlit interface. Users can rate
movies, get personalized recommendations, filter by genre, and read explanations
based on their rating history.

## Run

Use Python 3.10 or newer:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The command-line demo is also available:

```powershell
python main.py
```

Run the focused tests with:

```powershell
python -m unittest
```

The web app starts with sample profiles. Choose an existing profile or enter a
new name, then enter ratings from 1 to 5 in the movie table. Ratings and new
profiles persist in Streamlit's session state for the duration of that browser
session; they are not written to a database.

## Use your ratings

Create a pandas DataFrame with exactly one row per observed user/movie rating
and these columns:

| user_id | movie_id | rating |
| --- | --- | ---: |
| user_1 | 101 | 5 |
| user_2 | 101 | 3 |

Ratings must be from 1 to 5. Leave unrated movies out of the table; do not
encode missing ratings as `0`. Replace `sample_data()` in the script with your
loaded ratings and a dictionary mapping movie IDs to titles. Call
`evaluate_model(ratings)` to measure holdout RMSE/MAE, `train_model(ratings)`
to fit the final model, and `recommend_movies(...)` to get a DataFrame of
predicted ratings for movies the user has not rated. Use
`generate_recommendation_explanation(user_id, movie_id, ratings, metadata)` to
explain a recommendation using matching genres from metadata and the user's
high ratings. The explanation is deterministic: it only states genre matches
supported by the supplied data and otherwise discloses that the ranking comes
from rating patterns. The example metadata schema has `movie_id`, `title`, and
optional `genres` (a list or comma/pipe-separated string).

This example only recommends movies that have at least one rating in the
training data. It also requires known users; handling brand-new users needs an
onboarding or popularity-based fallback.
