import os

DATA_DIR = os.path.join(".", "data")
MODEL_DIR = os.path.join(".", "model")

RAW_DATA_PATH = os.path.join(DATA_DIR, "All_Beauty.json")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data")
MODEL_PATH = os.path.join(MODEL_DIR, "recommendation_model")
USER_CLUSTERS_PATH = os.path.join(DATA_DIR, "user_clusters")
PRODUCT_FEATURES_PATH = os.path.join(DATA_DIR, "product_features")

DEFAULT_ALS_PARAMS = {
    "rank": 50,
    "regParam": 0.1,
    "maxIter": 10,
    "alpha": 1.0,
    "nonnegative": True,
    "coldStartStrategy": "drop"
}

TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

SENTIMENT_POSITIVE_WORDS = [
    "great", "good", "excellent", "amazing", "love",
    "perfect", "best", "beautiful", "fantastic", "awesome"
]
SENTIMENT_NEGATIVE_WORDS = [
    "bad", "poor", "terrible", "worst", "hate",
    "disappointing", "awful", "horrible", "waste", "useless"
]

EVALUATION_METRICS = ["rmse", "mae"]

CLUSTER_COUNT = 5