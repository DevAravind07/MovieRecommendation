"""Interactive Streamlit interface for the movie recommendation demo."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from surprise import SVD

from main import (
    add_or_update_user_ratings,
    generate_recommendation_explanation,
    recommend_movies,
    sample_data,
    sample_movie_metadata,
    train_model,
)


@st.cache_resource(show_spinner="Updating recommendation model...")
def cached_model(ratings: pd.DataFrame) -> SVD:
    """Reuse a fitted model until the session's ratings data changes."""
    return train_model(ratings)


def _rating_editor_data(
    metadata: pd.DataFrame,
    ratings: pd.DataFrame,
    user_id: str | None,
) -> pd.DataFrame:
    editor_data = metadata.copy()
    editor_data["genres"] = editor_data["genres"].map(", ".join)
    existing_ratings = (
        ratings.loc[ratings["user_id"] == user_id]
        .set_index("movie_id")["rating"]
        if user_id is not None
        else pd.Series(dtype="float64")
    )
    editor_data["your_rating"] = editor_data["movie_id"].map(existing_ratings)
    return editor_data


def main() -> None:
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.title("🎬 Movie Recommender")
    st.write(
        "Rate movies to personalize your recommendations. Your submitted ratings "
        "are kept for this browser session."
    )

    if "ratings" not in st.session_state:
        st.session_state.ratings, _ = sample_data()
    ratings: pd.DataFrame = st.session_state.ratings
    metadata = sample_movie_metadata()
    base_ratings, _ = sample_data()
    existing_users = sorted(
        set(base_ratings["user_id"]) | set(ratings["user_id"]),
        key=str,
    )

    st.sidebar.header("Your profile")
    profile_mode = st.sidebar.radio(
        "Choose how to continue",
        ("Existing profile", "New profile"),
    )
    if profile_mode == "Existing profile":
        user_id = st.sidebar.selectbox("Profile", existing_users)
    else:
        entered_user_id = st.sidebar.text_input(
            "Choose a name or user ID",
            placeholder="e.g. kishore",
        ).strip()
        user_id = entered_user_id or None

    genres = sorted(
        set(metadata["genres"].explode().dropna().astype(str)),
        key=str.casefold,
    )
    selected_genres = st.sidebar.multiselect(
        "Filter recommendations by genre",
        genres,
        help="Leave empty to see recommendations from every genre.",
    )
    top_n = st.sidebar.slider("Number of recommendations", 1, 8, 5)

    st.subheader("Rate movies")
    st.caption(
        "Enter a whole-number rating from 1 (disliked) to 5 (loved). "
        "Existing profile ratings are shown and can be changed."
    )
    editor_data = _rating_editor_data(metadata, ratings, user_id)
    with st.form("movie_ratings"):
        edited_data = st.data_editor(
            editor_data,
            hide_index=True,
            width="stretch",
            disabled=["movie_id", "title", "genres"],
            column_config={
                "movie_id": None,
                "title": st.column_config.TextColumn("Movie"),
                "genres": st.column_config.TextColumn("Genres"),
                "your_rating": st.column_config.NumberColumn(
                    "Your rating",
                    min_value=1,
                    max_value=5,
                    step=1,
                    format="%d",
                ),
            },
            key="movie_rating_editor",
        )
        submitted = st.form_submit_button("Save ratings and recommend")

    if submitted:
        if user_id is None:
            st.error("Enter a name or user ID in the sidebar first.")
        else:
            submitted_ratings = edited_data.dropna(subset=["your_rating"])
            movie_ratings = dict(
                zip(submitted_ratings["movie_id"], submitted_ratings["your_rating"])
            )
            try:
                st.session_state.ratings = add_or_update_user_ratings(
                    st.session_state.ratings,
                    user_id,
                    movie_ratings,
                )
                st.success(f"Saved {len(movie_ratings)} rating(s) for {user_id}.")
                ratings = st.session_state.ratings
            except ValueError as error:
                st.error(str(error))

    if user_id is None:
        st.info("Choose an existing profile or enter a new profile name to continue.")
        return

    if not (ratings["user_id"] == user_id).any():
        st.info("Rate one or more movies above to start your personalized profile.")
        return

    st.subheader(f"Recommendations for {user_id}")
    model = cached_model(ratings)
    recommendations = recommend_movies(
        user_id,
        ratings,
        dict(zip(metadata["movie_id"], metadata["title"])),
        model,
        top_n=len(metadata),
    )
    if selected_genres:
        selected = {genre.casefold() for genre in selected_genres}
        recommendation_genres = metadata.set_index("movie_id")["genres"]
        recommendations = recommendations[
            recommendations["movie_id"].map(
                lambda movie_id: bool(
                    selected
                    & {
                        str(genre).casefold()
                        for genre in recommendation_genres.get(movie_id, [])
                    }
                )
            )
        ]
    recommendations = recommendations.head(top_n).reset_index(drop=True)

    if recommendations.empty:
        st.info(
            "No unrated movies match that genre filter. Change the filter or rate "
            "more movies."
        )
        return

    st.dataframe(
        recommendations.rename(
            columns={
                "movie_id": "Movie ID",
                "title": "Movie",
                "predicted_rating": "Predicted rating",
            }
        ).style.format({"Predicted rating": "{:.2f} / 5"}),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Why these movies?"):
        for recommendation in recommendations.itertuples(index=False):
            explanation = generate_recommendation_explanation(
                user_id,
                recommendation.movie_id,
                ratings,
                metadata,
            )
            st.markdown(f"- **{recommendation.title}:** {explanation}")


if __name__ == "__main__":
    main()
