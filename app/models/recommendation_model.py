from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("BookRecommendationEngine").getOrCreate()

# Load the trained model
model_path = "book_recommendation_model"
model = ALSModel.load(model_path)

def get_recommendations(user_id):
    user_df = spark.createDataFrame([(int(user_id),)], ["user_id"])
    recommendations = model.recommendForUserSubset(user_df, 5).collect()

    book_ids = [row.book_id for row in recommendations[0].recommendations]
    books_df = spark.read.csv('data/preprocessed_books.csv', header=True, inferSchema=True)
    book_details = books_df.filter(books_df.book_id.isin(book_ids)).collect()

    results = [{"title": book["Book-Title"], "author": book["Book-Author"]} for book in book_details]
    return results
