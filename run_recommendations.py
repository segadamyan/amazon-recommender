import argparse
import os

from pyspark.sql import SparkSession

import config as cfg
from evaluator import ModelEvaluator
from model_trainer import ModelTrainer
from recommendation_generator import RecommendationGenerator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Beauty Product Recommendation System")
    parser.add_argument("--mode", choices=["train", "recommend", "evaluate"],
                        default="train", help="Operation mode")
    parser.add_argument("--data_path", default=cfg.RAW_DATA_PATH,
                        help="Path to input data")
    parser.add_argument("--model_path", default=cfg.MODEL_PATH,
                        help="Path to save/load model")
    parser.add_argument("--user_id",
                        help="User ID for recommendations (required in recommend mode)")
    parser.add_argument("--num_recs", type=int, default=10,
                        help="Number of recommendations to generate")
    return parser.parse_args()


def init_spark():
    """Initialize Spark session"""
    return (SparkSession.builder
            .appName("Beauty_Recommendation_System")
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .getOrCreate())


def train_model(spark, data_path, model_path):
    """Train a new recommendation model"""
    from data_processor import DataProcessor
    from feature_engineering import FeatureEngineer


    processor = DataProcessor(spark)
    engineer = FeatureEngineer(spark)
    trainer = ModelTrainer(spark)

    print(f"Loading data from {data_path}")
    df_raw = spark.read.json(data_path)
    df_processed = processor.process_data(df_raw)
    df_features = engineer.create_features(df_processed)
    indexed_df = engineer.index_features(df_features)

    train, validation, test = indexed_df.randomSplit(
        [cfg.TRAIN_SPLIT, cfg.VALIDATION_SPLIT, cfg.TEST_SPLIT],
        seed=cfg.RANDOM_SEED
    )

    print("Training recommendation model...")
    best_model = trainer.tune_and_train(train, validation)

    evaluator = ModelEvaluator(spark)
    metrics = evaluator.evaluate_model(best_model, test)

    trainer.save_model(best_model, model_path)

    return best_model


def recommend(spark, model_path, user_id, num_recs):
    """Generate recommendations for a specific user"""

    trainer = ModelTrainer(spark)
    model = trainer.load_model(model_path)

    df = spark.read.parquet(cfg.PROCESSED_DATA_PATH)

    recommender = RecommendationGenerator(spark)

    try:
        user_id_num = float(user_id)
        recs = recommender.get_personalized_recommendations(model, user_id_num, df, num_recs)
    except ValueError:
        user_mapping = df.filter(df.reviewerID == user_id).select("userId").limit(1)

        if user_mapping.count() == 0:
            print(f"User {user_id} not found in the dataset.")
            return None

        user_id_num = user_mapping.first()["userId"]
        recs = recommender.get_personalized_recommendations(model, user_id_num, df, num_recs)

    if recs is not None:
        id_mapping = df.select("itemId", "asin").distinct()
        recs_with_ids = recs.join(id_mapping, on="itemId")

        print(f"\nTop {num_recs} recommendations for user {user_id}:")
        for rec in recs_with_ids.select("asin", "prediction").collect():
            print(f"Product {rec['asin']}: Predicted rating = {rec['prediction']:.2f}")

    return recs


def evaluate(spark, model_path):
    """Evaluate an existing model"""
    trainer = ModelTrainer(spark)
    model = trainer.load_model(model_path)

    if os.path.exists(cfg.PROCESSED_DATA_PATH):
        df = spark.read.parquet(cfg.PROCESSED_DATA_PATH)
        _, _, test = df.randomSplit([0.7, 0.15, 0.15], seed=cfg.RANDOM_SEED)
    else:
        print("Processed data not found. Please run training first.")
        return

    evaluator = ModelEvaluator(spark)
    metrics = evaluator.evaluate_model(model, test)

    evaluator.evaluate_by_user_group(model, test)

    return metrics


def main():
    """Main function to run the recommendation system"""
    args = parse_args()
    spark = init_spark()

    try:
        if args.mode == "train":
            train_model(spark, args.data_path, args.model_path)

        elif args.mode == "recommend":
            if not args.user_id:
                print("Error: User ID is required for recommendation mode.")
                return
            recommend(spark, args.model_path, args.user_id, args.num_recs)

        elif args.mode == "evaluate":
            evaluate(spark, args.model_path)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
