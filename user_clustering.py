from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.functions import avg, count


class UserClusterer:
    def __init__(self, spark):
        """Initialize the UserClusterer with a SparkSession"""
        self.spark = spark

    def build_user_clusters(self, df, output_path):
        """Cluster users based on their preferences and behaviors"""

        # Aggregate user features
        user_features = df.groupBy("userId").agg(
            avg("rating").alias("avg_rating"),
            avg("sentiment_score").alias("avg_sentiment"),
            avg("engagement_score").alias("avg_engagement"),
            count("rating").alias("review_count")
        )

        assembler = VectorAssembler(
            inputCols=["avg_rating", "avg_sentiment", "avg_engagement", "review_count"],
            outputCol="features"
        )

        scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

        pipeline = Pipeline(stages=[assembler, scaler])
        scaled_data = pipeline.fit(user_features).transform(user_features)

        wssse = []
        k_values = range(2, 11)

        for k in k_values:
            kmeans = KMeans(k=k, seed=42, featuresCol="scaled_features")
            model = kmeans.fit(scaled_data)
            wssse.append(model.summary.trainingCost)

        kmeans = KMeans(k=5, seed=42, featuresCol="scaled_features")
        model = kmeans.fit(scaled_data)

        user_clusters = model.transform(scaled_data)

        user_clusters.write.mode("overwrite").parquet(output_path)
        print(f"User clusters saved to {output_path}")

        cluster_counts = user_clusters.groupBy("prediction").count().orderBy("prediction")
        print("\nUser Cluster Distribution:")
        cluster_counts.show()

        print("\nCluster Centers:")
        for i, center in enumerate(model.clusterCenters()):
            print(f"Cluster {i}: {center}")

        return user_clusters, model

    def analyze_clusters(self, user_clusters, df):
        """Analyze the characteristics of each cluster"""

        df_with_clusters = df.join(
            user_clusters.select("userId", "prediction").withColumnRenamed("prediction", "cluster"),
            on="userId"
        )

        cluster_ratings = df_with_clusters.groupBy("cluster").agg(
            avg("rating").alias("avg_rating"),
            avg("sentiment_score").alias("avg_sentiment"),
            count("rating").alias("num_ratings")
        ).orderBy("cluster")

        print("\nCluster Characteristics:")
        cluster_ratings.show()

        return cluster_ratings