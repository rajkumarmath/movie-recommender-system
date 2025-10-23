# recommender.py
"""
MovieLens recommender using Spark MLlib (ALS).
Run: source venv/bin/activate && python recommender.py
"""

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")

def load_movie_titles(item_path):
    # u.item pipe-separated; first two fields: movieId|title
    titles = {}
    with open(item_path, encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                movie_id = int(parts[0])
                title = parts[1]
                titles[movie_id] = title
    return titles

def main():
    spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load ratings u.data: userId, movieId, rating, timestamp (tab separated)
    udata_path = os.path.join(DATA_DIR, "u.data")
    data = spark.read.csv(udata_path, sep="\t", inferSchema=True).toDF(
        "userId", "movieId", "rating", "timestamp"
    )

    # Basic stats
    print("Total ratings:", data.count())
    print("Users:", data.select("userId").distinct().count())
    print("Movies:", data.select("movieId").distinct().count())

    # Split
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    
    # Build ALS model
    als = ALS(
        maxIter=10,
        regParam=0.08,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=False
    )
    model = als.fit(train)

    # Evaluate
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE on test set: {rmse:.4f}")

    # Generate top-5 recommendations for all users
    user_recs = model.recommendForAllUsers(5)

    # Show sample
    user_recs.show(5, truncate=False)

    # Create a mapping of movieId -> title
    titles = load_movie_titles(os.path.join(DATA_DIR, "u.item"))

    # Example: get recommendations for a specific user (e.g., userId = 1)
    sample_user = 1
    user_rec_row = user_recs.filter(col("userId") == sample_user).collect()
    if user_rec_row:
        recs = user_rec_row[0]["recommendations"]
        print(f"\nTop recommendations for user {sample_user}:")
        for r in recs:
            mid = int(r["movieId"])
            score = float(r["rating"])
            print(f"  {titles.get(mid, str(mid))}  (score={score:.3f})")
    else:
        print(f"No recs found for user {sample_user}")

    # Optional: save model locally
    # model.write().overwrite().save("als_model")
    # Save model
    model.write().overwrite().save("movie_recommender_model")

    print("\nModel saved to: movie_recommender_model/")

    spark.stop()

if __name__ == "__main__":
    main()
