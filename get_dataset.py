import json
import os.path
import subprocess

import polars as pl

pl.Config.set_fmt_str_lengths(50)


def prepare_dataset():
    def movie_cast_as_text(cast_json, top_n=7):
        cast = json.loads(cast_json)
        formatted = ["- {role} played by {actor}. ".format(role=c["character"], actor=c["name"]) for c in cast[:top_n]]
        return "Cast:\n" + "\n".join(formatted)

    def movie_genres_as_text(genres_json, top_n=3):
        genres = json.loads(genres_json)
        formatted = "Movie genres:\n" + "\n".join(f"- {g['name'].lower()}" for g in genres[:top_n])
        return formatted

    # download raw dataset
    if not os.path.exists("kod/ml-workout-6-qa/TMDB-5000"):
        subprocess.run(["curl", "-O", "https://storage.ml-workout.pl/datasets/TMDB-5000.tar.bz2"])
        subprocess.run(["tar", "-xjf", "TMDB-5000.tar.bz2"])
    # read into polars dataframe
    credits = pl.read_csv("./TMDB-5000/tmdb_5000_credits.csv")
    movies = pl.read_csv("./TMDB-5000/tmdb_5000_movies.csv", infer_schema_length=10000)
    # join credits and movies
    tmdb = movies.select("title", "overview", "release_date", "genres", "id").join(
        credits.select("cast", "movie_id"), left_on="id", right_on="movie_id"
    )
    # preprocess
    # text dataframe consists of columns: [id, text]
    tmdb_texts = tmdb.select(
        "id",
        pl.struct(["title", "overview", "release_date", "genres", "cast"])
        .apply(
            lambda row: f"""passage: Title: {row['title']}
    Summary: {row['overview']}
    Released on: {row['release_date']}
    {movie_cast_as_text(row['cast'])}
    {movie_genres_as_text(row['genres'])}
    """.strip()
        )
        .alias("text"),
    )

    return {"text": tmdb_texts["text"].to_list(), "metadata": [{"id": text_id} for text_id in tmdb_texts["id"]]}
