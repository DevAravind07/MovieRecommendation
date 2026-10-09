"""Train and evaluate a small item-rating recommender using Surprise SVD."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

RATING_SCALE = (1, 5)
RATING_COLUMNS = ("user_id", "movie_id", "rating")


def sample_data() -> tuple[pd.DataFrame, dict[int, str]]:
    """Return a self-contained example ratings table and movie catalog."""
    rows = [
        ("user_1", 101, 5), ("user_1", 102, 4), ("user_1", 103, 5),
        ("user_1", 104, 2), ("user_1", 106, 1),
        ("user_2", 101, 4), ("user_2", 102, 5), ("user_2", 103, 4),
        ("user_2", 105, 2), ("user_2", 107, 1),
        ("user_3", 101, 1), ("user_3", 104, 5), ("user_3", 105, 4),
        ("user_3", 106, 5), ("user_3", 108, 4),
        ("user_4", 102, 2), ("user_4", 104, 4), ("user_4", 105, 5),
        ("user_4", 107, 4), ("user_4", 108, 5),
        ("user_5", 101, 5), ("user_5", 103, 4), ("user_5", 106, 4),
        ("user_5", 107, 2), ("user_5", 108, 3),
        ("user_6", 102, 4), ("user_6", 103, 5), ("user_6", 104, 1),
        ("user_6", 105, 2), ("user_6", 106, 5),
    ]
    ratings = pd.DataFrame(rows, columns=RATING_COLUMNS)
    titles = {
        101: "The Last Signal",
        102: "Autumn Harbor",
        103: "Beyond the Ridge",
        104: "Midnight Archive",
        105: "A Quiet Summer",
        106: "Orbit Seven",
        107: "Paper Kingdom",
        108: "The Glass Station",
    }
    return ratings, titles


def sample_movie_metadata() -> pd.DataFrame:
    """Return genres for the example catalog, for evidence-based explanations."""
    return pd.DataFrame(
        [
            (101, "The Last Signal", ["science fiction", "thriller"]),
            (102, "Autumn Harbor", ["drama", "romance"]),
            (103, "Beyond the Ridge", ["science fiction", "adventure"]),
            (104, "Midnight Archive", ["mystery", "thriller"]),
            (105, "A Quiet Summer", ["drama", "romance"]),
            (106, "Orbit Seven", ["science fiction", "adventure"]),
            (107, "Paper Kingdom", ["drama", "fantasy"]),
            (108, "The Glass Station", ["mystery", "science fiction"]),
        ],
        columns=("movie_id", "title", "genres"),
    )


def validate_ratings(ratings: pd.DataFrame) -> None:
    """Raise a clear error when the input cannot be used for training."""
    missing_columns = set(RATING_COLUMNS) - set(ratings.columns)
    if missing_columns:
        raise ValueError(f"Ratings are missing columns: {sorted(missing_columns)}")
    if ratings.empty:
        raise ValueError("Ratings must contain at least one row.")
    if ratings[list(RATING_COLUMNS)].isnull().any().any():
        raise ValueError("User IDs, movie IDs, and ratings cannot be empty.")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Each user/movie pair must have at most one rating.")

    numeric_ratings = pd.to_numeric(ratings["rating"], errors="coerce")
    if numeric_ratings.isnull().any() or not numeric_ratings.map(math.isfinite).all():
        raise ValueError("Ratings must be finite numbers.")
    low, high = RATING_SCALE
    if not numeric_ratings.between(low, high).all():
        raise ValueError(f"Ratings must be between {low} and {high}.")


def add_or_update_user_ratings(
    ratings: pd.DataFrame,
    user_id: Any,
    movie_ratings: Mapping[Any, Any],
) -> pd.DataFrame:
    """Return ratings with this user's submitted movie scores added or updated."""
    validate_ratings(ratings)
    if user_id is None or (isinstance(user_id, str) and not user_id.strip()):
        raise ValueError("User ID cannot be empty.")
    if not movie_ratings:
        raise ValueError("Submit at least one movie rating.")

    updated_rows = pd.DataFrame(
        [
            {"user_id": user_id, "movie_id": movie_id, "rating": rating}
            for movie_id, rating in movie_ratings.items()
        ]
    )
    numeric_ratings = pd.to_numeric(updated_rows["rating"], errors="coerce")
    if numeric_ratings.isnull().any() or not numeric_ratings.map(math.isfinite).all():
        raise ValueError("Ratings must be finite numbers.")
    low, high = RATING_SCALE
    if not numeric_ratings.between(low, high).all():
        raise ValueError(f"Ratings must be between {low} and {high}.")
    updated_rows["rating"] = numeric_ratings

    old_rows = ratings.loc[
        ~(
            (ratings["user_id"] == user_id)
            & ratings["movie_id"].isin(movie_ratings)
        )
    ]
    result = pd.concat([old_rows, updated_rows], ignore_index=True)
    validate_ratings(result)
    return result


def _genre_set(value: Any) -> set[str]:
    """Normalize common genre representations to comparable lowercase tags."""
    if isinstance(value, str):
        values = value.replace("|", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        return set()
    return {str(genre).strip().casefold() for genre in values if str(genre).strip()}


def generate_recommendation_explanation(
    user_id: Any,
    recommended_movie_id: Any,
    user_rating_history: pd.DataFrame,
    movie_metadata: pd.DataFrame,
) -> str:
    """Explain a recommendation using genres and titles present in the input data.

    The explanation is deterministic and does not claim that SVD used metadata:
    genre matches are surfaced as evidence, while the predicted preference is
    described as a rating-pattern result.
    """
    validate_ratings(user_rating_history)
    required_metadata = {"movie_id"}
    missing_columns = required_metadata - set(movie_metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Movie metadata is missing columns: {sorted(missing_columns)}"
        )
    if movie_metadata["movie_id"].isnull().any():
        raise ValueError("Movie metadata movie IDs cannot be empty.")
    if movie_metadata["movie_id"].duplicated().any():
        raise ValueError("Movie metadata must contain at most one row per movie.")

    user_history = user_rating_history[
        user_rating_history["user_id"] == user_id
    ].copy()
    if user_history.empty:
        raise ValueError(f"User {user_id!r} has no ratings to explain.")

    metadata_by_id = movie_metadata.set_index("movie_id", drop=False)
    target = (
        metadata_by_id.loc[recommended_movie_id]
        if recommended_movie_id in metadata_by_id.index
        else None
    )
    target_title = (
        str(target["title"])
        if target is not None and "title" in movie_metadata.columns
        else str(recommended_movie_id)
    )
    target_genres = (
        _genre_set(target.get("genres"))
        if target is not None and "genres" in movie_metadata.columns
        else set()
    )

    rated_ids = set(user_history["movie_id"])
    history_metadata = metadata_by_id[
        metadata_by_id.index.isin(rated_ids)
    ]
    history_by_id = history_metadata.to_dict(orient="index")
    history_rows = user_history.copy()
    history_rows["_rating"] = pd.to_numeric(history_rows["rating"])
    history_rows = history_rows.sort_values(
        "_rating",
        ascending=False,
        kind="stable",
    )

    if target_genres:
        for _, row in history_rows.iterrows():
            liked_metadata = history_by_id.get(row["movie_id"])
            if liked_metadata is None or row["_rating"] < 4:
                continue
            shared_genres = target_genres & _genre_set(liked_metadata.get("genres"))
            if shared_genres:
                liked_title = str(liked_metadata.get("title", row["movie_id"]))
                genres = ", ".join(sorted(shared_genres))
                return (
                    f"{target_title} is recommended because you rated "
                    f"{liked_title} {row['_rating']:g}/5, and both are tagged "
                    f"{genres}. Its ranking also reflects patterns learned "
                    f"from ratings in the dataset."
                )

    best_rating = history_rows.iloc[0]["_rating"]
    best_movie_id = history_rows.iloc[0]["movie_id"]
    best_metadata = history_by_id.get(best_movie_id, {})
    best_title = str(best_metadata.get("title", best_movie_id))
    return (
        f"{target_title} is recommended based on rating patterns in the dataset. "
        f"Your highest-rated movie so far is {best_title} "
        f"({best_rating:g}/5); no shared genre evidence was available."
    )


def _surprise_dataset(ratings: pd.DataFrame) -> Dataset:
    validate_ratings(ratings)
    reader = Reader(rating_scale=RATING_SCALE)
    dataset_frame = ratings.loc[:, list(RATING_COLUMNS)].copy()
    dataset_frame["rating"] = pd.to_numeric(dataset_frame["rating"])
    return Dataset.load_from_df(
        dataset_frame,
        reader,
    )


def evaluate_model(
    ratings: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return RMSE and MAE measured on a held-out ratings split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    data = _surprise_dataset(ratings)
    test_count = math.ceil(test_size * len(ratings))
    if test_count >= len(ratings):
        raise ValueError("Evaluation requires at least one training and one test rating.")
    trainset, testset = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )
    model = SVD(random_state=random_state)
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    return rmse, mae


def train_model(
    ratings: pd.DataFrame,
    *,
    random_state: int = 42,
) -> SVD:
    """Fit the final model using all available ratings."""
    data = _surprise_dataset(ratings)
    model = SVD(random_state=random_state)
    model.fit(data.build_full_trainset())
    return model


def recommend_movies(
    user_id: Any,
    ratings: pd.DataFrame,
    movie_titles: dict[Any, str],
    model: SVD,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank movies the user has not rated by predicted rating."""
    validate_ratings(ratings)
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if user_id not in set(ratings["user_id"]):
        raise ValueError(f"User {user_id!r} has no ratings; SVD cannot personalize them.")

    rated_movies = set(ratings.loc[ratings["user_id"] == user_id, "movie_id"])
    candidates = sorted(
        set(ratings["movie_id"]) - rated_movies,
        key=str,
    )
    predictions = [
        {
            "movie_id": movie_id,
            "title": movie_titles.get(movie_id, str(movie_id)),
            "predicted_rating": model.predict(user_id, movie_id).est,
        }
        for movie_id in candidates
    ]
    predictions.sort(key=lambda row: (-row["predicted_rating"], str(row["movie_id"])))
    return pd.DataFrame(
        predictions[:top_n],
        columns=("movie_id", "title", "predicted_rating"),
    )


def main() -> None:
    ratings, movie_titles = sample_data()
    movie_metadata = sample_movie_metadata()

    rmse, mae = evaluate_model(ratings)
    print(f"Holdout evaluation: RMSE={rmse:.3f}, MAE={mae:.3f}")

    model = train_model(ratings)
    user_id = "user_1"
    recommendations = recommend_movies(
        user_id,
        ratings,
        movie_titles,
        model,
        top_n=3,
    )
    print(f"\nTop recommendations for {user_id}:")
    if recommendations.empty:
        print("No unrated movies are available to recommend.")
    else:
        print(recommendations.to_string(index=False, formatters={
            "predicted_rating": "{:.2f}".format,
        }))
        print("\nWhy these movies?")
        for movie_id in recommendations["movie_id"]:
            print(
                "- "
                + generate_recommendation_explanation(
                    user_id,
                    movie_id,
                    ratings,
                    movie_metadata,
                )
            )


if __name__ == "__main__":
    main()
