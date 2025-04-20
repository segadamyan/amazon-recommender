import re

from pyspark.sql.functions import udf, col, length, split, when, size
from pyspark.sql.types import FloatType, ArrayType, StringType


class TextAnalyzer:
    def __init__(self, spark):
        """Initialize the TextAnalyzer with a SparkSession"""
        self.spark = spark
        self.positive_words = set([
            "great", "good", "excellent", "amazing", "love", "perfect", "best",
            "beautiful", "fantastic", "awesome", "wonderful", "happy", "impressed",
            "satisfied", "quality", "recommended", "favorite", "pleased", "outstanding",
            "superb", "fantastic", "delighted", "easy", "stunning", "brilliant"
        ])

        self.negative_words = set([
            "bad", "poor", "terrible", "worst", "hate", "disappointing", "awful",
            "horrible", "waste", "useless", "cheap", "broken", "defective", "problem",
            "issues", "return", "returned", "refund", "expensive", "avoid", "failed",
            "garbage", "junk", "frustrating", "difficult", "regret"
        ])

    def analyze_sentiment(self, df, text_col="reviewText"):
        """Perform sentiment analysis on text column"""

        def sentiment_score(text):
            if not text:
                return 0.0

            text = text.lower()
            words = re.findall(r'\b\w+\b', text)

            pos_count = sum(1 for word in words if word in self.positive_words)
            neg_count = sum(1 for word in words if word in self.negative_words)

            if pos_count + neg_count == 0:
                return 0.0

            return (pos_count - neg_count) / (pos_count + neg_count)

        sentiment_udf = udf(sentiment_score, FloatType())

        df = df.withColumn("sentiment_score", sentiment_udf(col(text_col)))

        df = df.withColumn(
            "sentiment_category",
            when(col("sentiment_score") > 0.3, "positive")
            .when(col("sentiment_score") < -0.3, "negative")
            .otherwise("neutral")
        )

        return df

    def extract_keywords(self, df, text_col="reviewText", n=5):
        """Extract top keywords from text"""

        stopwords = set([
            "the", "and", "is", "in", "it", "to", "for", "with", "on", "at", "of",
            "this", "that", "but", "was", "be", "are", "have", "has", "i", "a", "an",
            "they", "them", "their", "these", "those", "my", "your", "his", "her"
        ])

        def extract_top_words(text, n=n):
            if not text:
                return []

            text = text.lower()
            words = re.findall(r'\b\w+\b', text)

            filtered_words = [w for w in words if w not in stopwords and len(w) > 2]

            word_counts = {}
            for word in filtered_words:
                word_counts[word] = word_counts.get(word, 0) + 1

            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n]
            return [word for word, count in top_words]

        extract_keywords_udf = udf(extract_top_words, ArrayType(StringType()))

        # Apply keyword extraction
        df = df.withColumn("keywords", extract_keywords_udf(col(text_col)))

        return df

    def compute_text_features(self, df, text_col="reviewText"):
        """Compute various text features"""

        df = df.withColumn("text_length", length(col(text_col)))

        df = df.withColumn("word_count",
                           size(split(col(text_col), "\\s+")))

        def avg_word_length(text):
            if not text:
                return 0.0
            words = re.findall(r'\b\w+\b', text)
            if not words:
                return 0.0
            return sum(len(word) for word in words) / len(words)

        avg_word_udf = udf(avg_word_length, FloatType())
        df = df.withColumn("avg_word_length", avg_word_udf(col(text_col)))

        return df
