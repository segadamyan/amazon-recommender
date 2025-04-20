import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql.functions import col, count, desc, avg


def ensure_dir_exists(path):
    """Ensure that a directory exists, creating it if necessary"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def save_dataframe(df, path, format="parquet", mode="overwrite"):
    """Save a DataFrame to disk"""
    df.write.format(format).mode(mode).save(path)
    print(f"Saved DataFrame to {path}")


def load_dataframe(spark, path, format="parquet"):
    """Load a DataFrame from disk"""
    return spark.read.format(format).load(path)


def plot_rating_distribution(df, output_path=None):
    """Plot the distribution of ratings"""
    rating_counts = df.groupBy("rating").count().orderBy("rating").toPandas()

    plt.figure(figsize=(10, 6))
    plt.bar(rating_counts["rating"], rating_counts["count"])
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.xticks(np.arange(1, 6))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path)
        print(f"Saved rating distribution plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_temporal_trends(monthly_ratings, output_path=None):
    """Plot rating trends over time"""
    trends_df = monthly_ratings.toPandas()
    trends_df["date"] = pd.to_datetime(
        trends_df["year"].astype(str) + "-" + trends_df["month"].astype(str) + "-01"
    )
    trends_df = trends_df.sort_values("date")

    # Create plot
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot average rating
    ax1.plot(trends_df["date"], trends_df["avg_rating"], "b-", linewidth=2)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Average Rating", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim(1, 5)

    # Create second y-axis for review count
    ax2 = ax1.twinx()
    ax2.bar(trends_df["date"], trends_df["review_count"], alpha=0.3, color="r")
    ax2.set_ylabel("Review Count", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    plt.title("Rating Trends Over Time")
    fig.tight_layout()

    # Save or show
    if output_path:
        ensure_dir_exists(os.path.dirname(output_path))
        plt.savefig(output_path)
        print(f"Saved temporal trends plot to {output_path}")
    else:
        plt.show()

    plt.close()


def analyze_popular_products(df, n=10):
    """Analyze the most popular products"""
    # Get products with most reviews
    popular_products = df.groupBy("asin") \
        .agg(
        count("rating").alias("review_count"),
        avg("rating").alias("avg_rating")
    ) \
        .filter(col("review_count") > 5) \
        .orderBy(desc("review_count"))

    print(f"\nTop {n} Most Reviewed Products:")
    popular_products.limit(n).show()

    # Get highest rated products (with minimum reviews)
    top_rated = df.groupBy("asin") \
        .agg(
        count("rating").alias("review_count"),
        avg("rating").alias("avg_rating")
    ) \
        .filter(col("review_count") > 10) \
        .orderBy(desc("avg_rating"), desc("review_count"))

    print(f"\nTop {n} Highest Rated Products (min 10 reviews):")
    top_rated.limit(n).show()

    return popular_products, top_rated


def create_summary_report(df, metrics, output_path):
    """Create a summary report of the recommendation system"""
    # Basic stats
    total_users = df.select("userId").distinct().count()
    total_items = df.select("itemId").distinct().count()
    total_ratings = df.count()
    avg_rating = df.agg(avg("rating")).first()[0]

    # Create report
    with open(output_path, "w") as f:
        f.write("# Beauty Product Recommendation System Report\n\n")

        f.write("## Dataset Statistics\n")
        f.write(f"- Total Users: {total_users}\n")
        f.write(f"- Total Products: {total_items}\n")
        f.write(f"- Total Ratings: {total_ratings}\n")
        f.write(f"- Average Rating: {avg_rating:.2f}\n\n")

        f.write("## Model Performance\n")
        f.write(f"- RMSE: {metrics[0]:.4f}\n")
        f.write(f"- MAE: {metrics[1]:.4f}\n")
        f.write(f"- Coverage: {metrics[2]:.4f}\n")
        f.write(f"- Precision@k: {metrics[3]:.4f}\n\n")

        f.write("## Recommendations\n")
        f.write(
            "The system provides personalized recommendations based on user preferences and item characteristics.\n")
        f.write("It combines collaborative filtering with content analysis for improved accuracy.\n")