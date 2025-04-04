from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder.appName("BookRecommendationEngine").getOrCreate()

books_path = 'data/Books.csv'
ratings_path = 'data/Ratings.csv'
users_path = 'data/Users.csv'

books_df = spark.read.csv(books_path, header=True, inferSchema=True)
ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)
users_df = spark.read.csv(users_path, header=True, inferSchema=True)

book_indexer = StringIndexer(inputCol="ISBN", outputCol="book_id")
user_indexer = StringIndexer(inputCol="User-ID", outputCol="user_id")

books_df = book_indexer.fit(books_df).transform(books_df)
ratings_df = user_indexer.fit(ratings_df).transform(ratings_df)

books_df.write.csv('data/preprocessed_books.csv', header=True)
ratings_df.write.csv('data/preprocessed_ratings.csv', header=True)
users_df.write.csv('data/preprocessed_users.csv', header=True)
