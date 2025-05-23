import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import month, count, avg, year

from data_processor import DataProcessor
from evaluator import ModelEvaluator
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from recommendation_generator import RecommendationGenerator
from user_clustering import UserClusterer
from utils import plot_rating_distribution, plot_temporal_trends, plot_avg_rating_by_verified, \
    plot_review_length_distribution

DATA_PATH = os.path.join(".", "data", "All_Beauty.json")
PROCESSED_DATA_PATH = os.path.join(".", "data", "processed_data")
MODEL_PATH = os.path.join(".", "model", "recommendation_model")
USER_CLUSTERS_PATH = os.path.join(".", "data", "user_clusters")


def main():
    print("Starting Beauty Product Recommendation System")
    start_time = time.time()

    spark = (SparkSession.builder
             .appName("Advanced_Beauty_Recommendation")
             .getOrCreate())

    spark.conf.set("spark.sql.adaptive.enabled", "true")

    data_processor = DataProcessor(spark)
    feature_engineer = FeatureEngineer(spark)
    model_trainer = ModelTrainer(spark)
    evaluator = ModelEvaluator(spark)
    recommender = RecommendationGenerator(spark)
    clusterer = UserClusterer(spark)

    print("Loading and preprocessing data...")
    df_raw = spark.read.json(DATA_PATH)
    df_processed = data_processor.process_data(df_raw)

    df_with_features = feature_engineer.create_features(df_processed)

    indexed_df = feature_engineer.index_features(df_with_features)

    plot_rating_distribution(df_with_features, "plots/rating_distribution.png")

    monthly_ratings = df_processed.withColumn("year", year("reviewDate")) \
        .withColumn("month", month("reviewDate")) \
        .groupBy("year", "month") \
        .agg(
        avg("rating").alias("avg_rating"),
        count("rating").alias("review_count")
    )

    plot_temporal_trends(monthly_ratings, "plots/temporal_trends.png")
    plot_avg_rating_by_verified(df_with_features, "plots/avg_rating_by_verified.png")
    plot_review_length_distribution(df_with_features, "plots/review_length_distribution.png")

    indexed_df.write.mode("overwrite").parquet(PROCESSED_DATA_PATH)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")

    clusterer.build_user_clusters(indexed_df, USER_CLUSTERS_PATH)

    train, validation, test = indexed_df.randomSplit([0.7, 0.15, 0.15], seed=42)

    best_model = model_trainer.train_basic_model(train)

    evaluator.evaluate_model(best_model, test)

    recommender.generate_sample_recommendations(best_model, indexed_df)

    recommender.simulate_real_time_recommendations(best_model, indexed_df)

    # Save the model
    best_model.write().overwrite().save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
    spark.stop()


if __name__ == "__main__":
    main()