# app.py
from flask import Flask, request, jsonify
import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel  # <-- this line fixes the error

app = Flask(__name__)

# === Load movie titles from u.item ===
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")
titles = {}
with open(os.path.join(DATA_DIR, "u.item"), encoding="ISO-8859-1") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) >= 2:
            titles[int(parts[0])] = parts[1]

# === Spark setup ===
spark = SparkSession.builder.appName("FlaskRecommender").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# === Try to load saved model; if not found, train one ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "movie_recommender_model")

def get_model():
    if os.path.exists(MODEL_PATH):
        print("Loading saved ALS model...")
        return ALSModel.load(MODEL_PATH)
    else:
        print("Training new ALS model (first time only)...")
        data = spark.read.csv(os.path.join(DATA_DIR, "u.data"), sep="\t", inferSchema=True).toDF(
            "userId", "movieId", "rating", "timestamp"
        )
        train = data.sample(False, 0.8, seed=42)
        als = ALS(
            maxIter=8,
            regParam=0.1,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True
        )
        model = als.fit(train)
        model.write().overwrite().save(MODEL_PATH)
        return model

model = get_model()

# === Flask route ===
@app.route("/recommend")
def recommend():
    uid = int(request.args.get("user", 1))
    rec_df = model.recommendForUserSubset(spark.createDataFrame([(uid,)], ["userId"]), 5)
    rows = rec_df.collect()
    if not rows:
        return jsonify({"user": uid, "recommendations": []})
    recs = rows[0]["recommendations"]
    result = []
    for r in recs:
        mid = int(r["movieId"])
        result.append({
            "movieId": mid,
            "title": titles.get(mid, "Unknown"),
            "score": float(r["rating"])
        })
    return jsonify({"user": uid, "recommendations": result})

if __name__ == "__main__":
    app.run(port=5000)
