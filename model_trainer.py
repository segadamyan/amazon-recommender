from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator


class ModelTrainer:
    def __init__(self, spark):
        """Initialize the ModelTrainer with a SparkSession"""
        self.spark = spark

    def tune_and_train(self, train, validation):
        """Tune ALS model parameters using cross-validation"""

        als = ALS(
            userCol="userId",
            itemCol="itemId",
            ratingCol="weighted_rating",
            coldStartStrategy="drop",
            nonnegative=True,
            implicitPrefs=False,
        )

        param_grid = (
            ParamGridBuilder()
            .addGrid(als.rank, [10, 50, 100])
            .addGrid(als.regParam, [0.01, 0.1, 1.0])
            .addGrid(als.maxIter, [10, 20])
            .addGrid(als.alpha, [0.01, 1.0, 40.0])
            .build()
        )

        evaluator = RegressionEvaluator(
            metricName="rmse", labelCol="weighted_rating", predictionCol="prediction"
        )

        cv = CrossValidator(
            estimator=als,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
        )

        print("Performing hyperparameter tuning (this may take some time)...")
        cv_model = cv.fit(train)

        best_model = cv_model.bestModel

        print(f"Best Model Parameters:")
        print(f"Rank: {best_model.getRank()}")
        print(f"RegParam: {best_model.getRegParam()}")
        print(f"MaxIter: {best_model.getMaxIter()}")
        print(f"Alpha: {best_model._java_obj.getAlpha()}")

        return best_model

    def train_basic_model(self, train_data):
        """Train a basic ALS model with default parameters"""

        als = ALS(
            userCol="userId",
            itemCol="itemId",
            ratingCol="weighted_rating",
            coldStartStrategy="drop",
            nonnegative=True,
            rank=50,
            regParam=0.1,
            maxIter=10,
            alpha=1.0,
        )

        model = als.fit(train_data)
        return model

    def save_model(self, model, path):
        """Save the trained model to disk"""
        model.write().overwrite().save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a previously trained model from disk"""
        from pyspark.ml.recommendation import ALSModel

        model = ALSModel.load(path)
        return model
