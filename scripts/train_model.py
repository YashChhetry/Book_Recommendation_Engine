from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName("BookRecommendationEngine").getOrCreate()

ratings_df = spark.read.csv('data/preprocessed_ratings.csv', header=True, inferSchema=True)

als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="book_id", ratingCol="Book-Rating",
          coldStartStrategy="drop", nonnegative=True)
model = als.fit(ratings_df)
model.save("models/book_recommendation_model")
