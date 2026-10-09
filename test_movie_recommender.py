import unittest

from main import (
    add_or_update_user_ratings,
    generate_recommendation_explanation,
    recommend_movies,
    sample_data,
    sample_movie_metadata,
    train_model,
    validate_ratings,
)


class MovieRecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ratings, cls.movie_titles = sample_data()
        cls.model = train_model(cls.ratings)

    def test_recommendations_exclude_movies_already_rated(self):
        recommendations = recommend_movies(
            "user_1",
            self.ratings,
            self.movie_titles,
            self.model,
            top_n=3,
        )
        rated_movies = set(
            self.ratings.loc[self.ratings["user_id"] == "user_1", "movie_id"]
        )
        self.assertEqual(len(recommendations), 3)
        self.assertTrue(set(recommendations["movie_id"]).isdisjoint(rated_movies))
        self.assertTrue(recommendations["predicted_rating"].is_monotonic_decreasing)

    def test_unknown_user_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "has no ratings"):
            recommend_movies(
                "new_user",
                self.ratings,
                self.movie_titles,
                self.model,
            )

    def test_duplicate_user_movie_ratings_are_rejected(self):
        duplicate_ratings = self.ratings.iloc[:2].copy()
        duplicate_ratings.loc[1, "movie_id"] = duplicate_ratings.iloc[0]["movie_id"]
        with self.assertRaisesRegex(ValueError, "at most one rating"):
            validate_ratings(duplicate_ratings)

    def test_explanation_uses_shared_genre_and_high_rating(self):
        explanation = generate_recommendation_explanation(
            "user_1",
            105,
            self.ratings,
            sample_movie_metadata(),
        )
        self.assertIn("A Quiet Summer", explanation)
        self.assertIn("Autumn Harbor 4/5", explanation)
        self.assertIn("romance", explanation)

    def test_explanation_is_transparent_when_genres_are_unavailable(self):
        metadata_without_genres = sample_movie_metadata()[["movie_id", "title"]]
        explanation = generate_recommendation_explanation(
            "user_1",
            105,
            self.ratings,
            metadata_without_genres,
        )
        self.assertIn("rating patterns in the dataset", explanation)
        self.assertIn("no shared genre evidence was available", explanation)

    def test_explanation_rejects_user_without_history(self):
        with self.assertRaisesRegex(ValueError, "has no ratings to explain"):
            generate_recommendation_explanation(
                "new_user",
                105,
                self.ratings,
                sample_movie_metadata(),
            )

    def test_new_user_ratings_are_added(self):
        updated = add_or_update_user_ratings(
            self.ratings,
            "new_user",
            {101: 5, 104: 2},
        )
        new_user_ratings = updated[updated["user_id"] == "new_user"]
        self.assertEqual(len(new_user_ratings), 2)
        self.assertEqual(
            dict(zip(new_user_ratings["movie_id"], new_user_ratings["rating"])),
            {101: 5, 104: 2},
        )

    def test_existing_user_rating_is_updated_without_duplicate(self):
        updated = add_or_update_user_ratings(
            self.ratings,
            "user_1",
            {101: 3},
        )
        current_rating = updated.loc[
            (updated["user_id"] == "user_1") & (updated["movie_id"] == 101),
            "rating",
        ]
        self.assertEqual(len(current_rating), 1)
        self.assertEqual(current_rating.iloc[0], 3)

    def test_empty_user_rating_submission_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one movie rating"):
            add_or_update_user_ratings(self.ratings, "new_user", {})


if __name__ == "__main__":
    unittest.main()
