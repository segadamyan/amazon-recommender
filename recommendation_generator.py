from pyspark.sql.functions import col, avg, count


class RecommendationGenerator:
    def __init__(self, spark):
        """Initialize the RecommendationGenerator with a SparkSession"""
        self.spark = spark

    def generate_sample_recommendations(self, model, df):
        """Generate recommendations for sample users"""

        active_users = (
            df.groupBy("userId")
            .agg(count("rating").alias("review_count"))
            .orderBy(col("review_count").desc())
            .limit(5)
        )

        active_user_ids = [row["userId"] for row in active_users.collect()]
        user_recs = model.recommendForUserSubset(
            df.select("userId").filter(col("userId").isin(active_user_ids)).distinct(),
            10,
        )

        print("\nSample Recommendations for Active Users:")
        for user_rec in user_recs.collect():
            user_id = user_rec["userId"]
            items = user_rec["recommendations"]
            print(f"User {user_id}: {[item['itemId'] for item in items]}")

        id_mapping = df.select("itemId", "asin").distinct()

        item_to_product = {row["itemId"]: row["asin"] for row in id_mapping.collect()}

        print("\nRecommendations with Product IDs:")
        for user_rec in user_recs.collect():
            user_id = user_rec["userId"]
            items = user_rec["recommendations"]
            product_ids = [
                item_to_product.get(item["itemId"], "unknown") for item in items
            ]
            print(f"User {user_id}: {product_ids[:3]} ...")

        return user_recs

    def get_top_items(self, df, n=10):
        """Get top rated items for cold-start scenarios"""

        item_avgs = (
            df.groupBy("itemId", "asin")
            .agg(
                avg("weighted_rating").alias("avg_rating"),
                count("rating").alias("count"),
            )
            .filter(col("count") > 5)
            .orderBy(col("avg_rating").desc())
            .limit(n)
        )

        print("\nTop Items for Cold-Start Users:")
        for item in item_avgs.collect():
            print(
                f"Item {item['asin']}: Average Rating = {item['avg_rating']:.2f} (from {item['count']} reviews)"
            )

        return item_avgs

    def simulate_real_time_recommendations(self, model, df):
        """Simulate real-time recommendation scenarios"""

        sample_user = df.select("userId", "reviewerID").distinct().limit(1).collect()[0]
        user_id = sample_user["userId"]
        user_original_id = sample_user["reviewerID"]

        print(f"\nReal-time recommendations for user {user_original_id}:")

        user_items = set(
            [
                row["itemId"]
                for row in df.filter(col("userId") == user_id)
                .select("itemId")
                .collect()
            ]
        )
        all_items = set(
            [
                row["itemId"]
                for row in df.select("itemId").distinct().limit(1000).collect()
            ]
        )
        candidate_items = list(all_items - user_items)

        if len(candidate_items) > 100:
            candidate_items = candidate_items[:100]

        predict_df = self.spark.createDataFrame(
            [(user_id, item_id) for item_id in candidate_items], ["userId", "itemId"]
        )

        predictions = model.transform(predict_df)

        top_recs = predictions.orderBy(col("prediction").desc()).limit(5)

        id_mapping = df.select("itemId", "asin").distinct()
        top_recs_with_ids = top_recs.join(id_mapping, on="itemId")

        print("\nTop recommendations for this user:")
        for rec in top_recs_with_ids.select("asin", "prediction").collect():
            print(f"Product {rec['asin']}: Predicted rating = {rec['prediction']:.2f}")

        return top_recs_with_ids

    def get_personalized_recommendations(self, model, user_id, df, n=10):
        """Get personalized recommendations for a specific user"""

        user_features = df.filter(col("userId") == user_id)

        if user_features.count() == 0:
            print(f"User {user_id} not found.")
            return None

        user_items = set(
            [row["itemId"] for row in user_features.select("itemId").collect()]
        )
        all_items = set(
            [row["itemId"] for row in df.select("itemId").distinct().collect()]
        )
        candidate_items = list(all_items - user_items)

        predict_df = self.spark.createDataFrame(
            [(user_id, item_id) for item_id in candidate_items], ["userId", "itemId"]
        )

        predictions = model.transform(predict_df)
        top_n = predictions.orderBy(col("prediction").desc()).limit(n)

        return top_n
