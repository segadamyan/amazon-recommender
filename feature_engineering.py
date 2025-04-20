from pyspark.sql.functions import col, count, avg, stddev, when
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer


class FeatureEngineer:
    def __init__(self, spark):
        """Initialize the FeatureEngineer with a SparkSession"""
        self.spark = spark

    def create_features(self, df):
        """Create advanced features for better recommendations"""

        product_popularity = df.groupBy("asin") \
            .agg(
            count("rating").alias("review_count"),
            avg("rating").alias("avg_rating"),
            stddev("rating").alias("rating_stddev")
        )

        user_engagement = df.groupBy("reviewerID") \
            .agg(
            count("rating").alias("user_review_count"),
            avg("rating").alias("user_avg_rating"),
            stddev("rating").alias("user_rating_stddev")
        )

        df = df.join(product_popularity, on="asin", how="left")
        df = df.join(user_engagement, on="reviewerID", how="left")

        # Calculate Z-score for each rating to identify unusual ratings
        window_user = Window.partitionBy("reviewerID")
        df = df.withColumn("user_zscore",
                           (col("rating") - col("user_avg_rating")) /
                           when(col("user_rating_stddev").isNull() | (col("user_rating_stddev") == 0), 1)
                           .otherwise(col("user_rating_stddev")))

        df = df.withColumn("engagement_score",
                           col("review_length") / 100 * 0.3 +
                           col("user_review_count") / 10 * 0.3 +
                           (1 - (col("recency") / 365)) * 0.4)

        df = df.fillna({
            "rating_stddev": 0,
            "user_rating_stddev": 0,
            "engagement_score": 0.5,
            "sentiment_score": 0
        })

        return df

    def index_features(self, df):
        """Index categorical features and prepare for model training"""

        # Index users and items
        user_indexer = StringIndexer(inputCol="reviewerID", outputCol="userId").fit(df)
        item_indexer = StringIndexer(inputCol="asin", outputCol="itemId").fit(df)

        indexed_df = user_indexer.transform(df)
        indexed_df = item_indexer.transform(indexed_df)

        model_df = indexed_df.select(
            "userId", "itemId", "rating", "weighted_rating",
            "sentiment_score", "engagement_score", "user_zscore",
            "review_count", "user_review_count",
            "reviewerID", "asin"
        )

        return model_df