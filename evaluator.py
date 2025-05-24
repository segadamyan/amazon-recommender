from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when, avg


class ModelEvaluator:
    def __init__(self, spark):
        """Initialize the ModelEvaluator with a SparkSession"""
        self.spark = spark

    def evaluate_model(self, model, test):
        """Evaluate model on test data using multiple metrics"""

        predictions = model.transform(test)

        evaluator = RegressionEvaluator(
            metricName="rmse", labelCol="weighted_rating", predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)

        evaluator = RegressionEvaluator(
            metricName="mae", labelCol="weighted_rating", predictionCol="prediction"
        )
        mae = evaluator.evaluate(predictions)

        total_pairs = test.count()
        predicted_pairs = predictions.filter(~col("prediction").isNull()).count()
        coverage = predicted_pairs / total_pairs if total_pairs > 0 else 0

        k_predictions = predictions.withColumn(
            "correct_prediction",
            when(
                (col("prediction") >= 4.0) & (col("weighted_rating") >= 4.0), 1.0
            ).otherwise(0.0),
        )

        precision = k_predictions.agg(avg("correct_prediction")).first()[0]

        print("\nModel Evaluation:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Precision@k: {precision:.4f}")

        return rmse, mae, coverage, precision

    def evaluate_by_user_group(self, model, test):
        """Evaluate model performance across different user segments"""

        user_activity = (
            test.groupBy("userId")
            .count()
            .withColumn(
                "activity_level",
                when(col("count") < 5, "low")
                .when(col("count") < 20, "medium")
                .otherwise("high"),
            )
        )

        test_with_activity = test.join(
            user_activity.select("userId", "activity_level"), on="userId"
        )

        predictions = model.transform(test_with_activity)

        metrics_by_group = []

        for group in ["low", "medium", "high"]:
            group_predictions = predictions.filter(col("activity_level") == group)

            if group_predictions.count() > 0:
                evaluator = RegressionEvaluator(
                    metricName="rmse",
                    labelCol="weighted_rating",
                    predictionCol="prediction",
                )
                rmse = evaluator.evaluate(group_predictions)

                # Calculate coverage
                total = group_predictions.count()
                predicted = group_predictions.filter(
                    ~col("prediction").isNull()
                ).count()
                coverage = predicted / total if total > 0 else 0

                metrics_by_group.append((group, rmse, coverage))

        print("\nEvaluation by User Activity Level:")
        for group, rmse, coverage in metrics_by_group:
            print(f"Group: {group}, RMSE: {rmse:.4f}, Coverage: {coverage:.4f}")

        return metrics_by_group
