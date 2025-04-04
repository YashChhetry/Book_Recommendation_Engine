from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# Initialize Flask application
app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName('RecommendationEngine').getOrCreate()

# Load the trained ALS model
model_path = 'models\book_recommendation_model'
model = ALSModel.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    
    # Create a DataFrame for the user_id
    user_df = spark.createDataFrame([(user_id,)], ["user_id"])
    
    # Get recommendations for the user
    user_recommendations = model.recommendForUserSubset(user_df, 10).collect()
    
    recommendations = []
    if user_recommendations:
        for row in user_recommendations[0]['recommendations']:
            recommendations.append({'anime_id': row['anime_id'], 'rating': row['rating']})

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
