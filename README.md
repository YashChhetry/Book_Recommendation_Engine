**Requirements:**
Flask==2.0.1
pyspark==3.1.2
pandas==1.3.3

To install pyspark:
pip install pyspark


**Code:**
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName("BookRecommendationEngine").getOrCreate()

books_path = '/content/drive/MyDrive/data/Books.csv'
ratings_path = '/content/drive/MyDrive/data/Ratings.csv'
users_path = '/content/drive/MyDrive/data/Users.csv'

**Load data into DataFrames**
books_df = spark.read.csv(books_path, header=True, inferSchema=True)
ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)
users_df = spark.read.csv(users_path, header=True, inferSchema=True)

**Index the books and users**
book_indexer = StringIndexer(inputCol="ISBN", outputCol="book_id")
books_df = book_indexer.fit(books_df).transform(books_df)

user_indexer = StringIndexer(inputCol="User-ID", outputCol="user_id")
ratings_df = user_indexer.fit(ratings_df).transform(ratings_df)

ratings_df = ratings_df.join(books_df.select("ISBN", "book_id"), on="ISBN", how="inner")

**# Initialize ALS model**
als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="book_id", ratingCol="Book-Rating", coldStartStrategy="drop", nonnegative=True)
model = als.fit(ratings_df)

model.save("/content/drive/MyDrive/book_recommendation.model")
