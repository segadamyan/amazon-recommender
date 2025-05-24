import re
import textstat

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    lower,
    split,
    size,
    array_intersect,
    lit,
    when,
    length,
    col,
    udf,
)
from pyspark.sql.types import FloatType, ArrayType, StringType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    CountVectorizerModel,
)

from nltk.corpus import opinion_lexicon


class TextAnalyzer:
    def __init__(
        self, spark, text_col="reviewText", vocab_size=10000, top_n_keywords=5
    ):
        """Initialize Spark context, broadcast lexicons, and build ML pipeline blocks."""
        self.spark = spark
        sc = spark.sparkContext

        self.pos_bc = sc.broadcast(opinion_lexicon.positive())
        self.neg_bc = sc.broadcast(opinion_lexicon.negative())

        # --- keyword extraction pipeline pieces ---
        self.text_col = text_col
        self.top_n = top_n_keywords

        tokenizer = RegexTokenizer(
            inputCol=text_col, outputCol="tokens", pattern="\\W+"
        )
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
        cvector = CountVectorizer(
            inputCol="filtered", outputCol="rawFeatures", vocabSize=vocab_size
        )
        idf = IDF(inputCol="rawFeatures", outputCol="tfidfFeatures")

        self.kw_pipeline = Pipeline(stages=[tokenizer, remover, cvector, idf])

    def analyze_sentiment(self, df):
        """Fast Spark-native sentiment_score & category via broadcast + array_intersect."""
        df = (
            df.withColumn("words", split(lower(col(self.text_col)), "\\W+"))
            .withColumn(
                "pos_count", size(array_intersect(col("words"), lit(self.pos_bc.value)))
            )
            .withColumn(
                "neg_count", size(array_intersect(col("words"), lit(self.neg_bc.value)))
            )
            .withColumn(
                "sentiment_score",
                (col("pos_count") - col("neg_count"))
                / when((col("pos_count") + col("neg_count")) == 0, lit(1)).otherwise(
                    col("pos_count") + col("neg_count")
                ),
            )
            .withColumn(
                "sentiment_category",
                when(col("sentiment_score") > 0.3, "positive")
                .when(col("sentiment_score") < -0.3, "negative")
                .otherwise("neutral"),
            )
        )
        return df

    def extract_keywords(self, df):
        """TF–IDF pipeline + top-N vocabulary terms per document."""
        model = self.kw_pipeline.fit(df)
        df = model.transform(df)

        # pull out vocabulary from CountVectorizerModel stage
        cv_model: CountVectorizerModel = model.stages[2]
        vocab = cv_model.vocabulary

        def top_n(raw: "SparseVector"):
            if raw is None:
                return []
            # raw.indices and raw.values correspond
            pairs = list(zip(raw.indices, raw.values))
            topk = sorted(pairs, key=lambda x: x[1], reverse=True)[: self.top_n]
            return [vocab[i] for i, _ in topk]

        top_n_udf = udf(top_n, ArrayType(StringType()))
        return df.withColumn("keywords", top_n_udf(col("rawFeatures")))

    def compute_text_features(self, df):
        """Various extra text features: length, word_count, avg_word_len, sentences,
        punctuation ratio, readability."""
        # basic counts
        df = df.withColumn("text_length", length(col(self.text_col))).withColumn(
            "word_count", size(split(col(self.text_col), "\\s+"))
        )

        # average word length
        def avg_len(text):
            words = re.findall(r"\b\w+\b", text or "")
            return float(sum(len(w) for w in words) / len(words)) if words else 0.0

        df = df.withColumn(
            "avg_word_length", udf(avg_len, FloatType())(col(self.text_col))
        )

        # sentence count
        df = df.withColumn("sentence_count", size(split(col(self.text_col), "[\\.!?]")))

        # punctuation ratio
        def punct_ratio(text):
            if not text:
                return 0.0
            total_words = len(text.split())
            punct_marks = sum(ch in "!?." for ch in text)
            return float(punct_marks) / total_words if total_words else 0.0

        df = df.withColumn(
            "punctuation_ratio", udf(punct_ratio, FloatType())(col(self.text_col))
        )

        # readability (Flesch–Kincaid Grade via textstat)
        def fk_grade(text):
            try:
                return float(textstat.flesch_kincaid_grade(text))
            except:
                return None

        df = df.withColumn(
            "readability_grade", udf(fk_grade, FloatType())(col(self.text_col))
        )

        return df

    def transform(self, df):
        """Run the full flow and return all features + improved sentiment."""
        df = self.analyze_sentiment(df)
        df = self.extract_keywords(df)
        df = self.compute_text_features(df)
        return df
