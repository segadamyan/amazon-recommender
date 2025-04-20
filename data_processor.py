from pyspark.sql.functions import col, length, from_unixtime, datediff, current_date, when


class DataProcessor:
    def __init__(self, spark):
        """Initialize the DataProcessor with a SparkSession"""
        self.spark = spark

    def process_data(self, df_raw):
        """Process raw data with advanced cleaning and feature extraction"""
        df = df_raw.select(
            "reviewerID",
            "asin",
            col("overall").cast("float").alias("rating"),
            "reviewText",
            "summary",
            "unixReviewTime",
            "verified"
        )

        df = df.filter(df.reviewText.isNotNull() & (length(df.reviewText) > 5))

        df = df.withColumn("reviewDate", from_unixtime(col("unixReviewTime")).cast("date"))

        df = df.withColumn("recency", datediff(current_date(), col("reviewDate")))

        df = df.withColumn("verified", col("verified").cast("boolean"))
        df = df.withColumn("verified_weight", when(col("verified") == True, 1.2).otherwise(0.8))

        df = df.withColumn("weighted_rating", col("rating") * col("verified_weight"))

        df = self.analyze_review_text(df)
        return df

    def analyze_review_text(self, df):
        df = df.withColumn("review_length", length(col("reviewText")))

        positive_words = ["great", "good", "excellent", "amazing", "love", "perfect", "best"]
        negative_words = ["bad", "poor", "terrible", "worst", "hate", "disappointing", "awful"]

        from pyspark.sql.types import FloatType
        from pyspark.sql.functions import udf

        def basic_sentiment(text):
            if text is None:
                return 0.0
            text = text.lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            if pos_count + neg_count == 0:
                return 0.0
            return (pos_count - neg_count) / (pos_count + neg_count)

        sentiment_udf = udf(basic_sentiment, FloatType())
        df = df.withColumn("sentiment_score", sentiment_udf(col("reviewText")))

        return df

    def load_and_preprocess(self, file_path):
        """Convenience method to load and preprocess in one step"""
        df_raw = self.spark.read.json(file_path)
        return self.process_data(df_raw)