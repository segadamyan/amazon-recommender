# Advanced Beauty Product Recommendation System

An advanced recommendation system built with PySpark for the Amazon Beauty Products dataset.

## Project Structure

```
recommendation_system/
├── main.py                      # Main entry point
├── run_recommendations.py       # Production runner script
├── config.py                    # Configuration settings
├── data_processor.py            # Data loading and preprocessing
├── feature_engineering.py       # Feature creation and indexing
├── model_trainer.py             # ALS model training and tuning
├── evaluator.py                 # Model evaluation module 
├── recommendation_generator.py  # Recommendation generation
├── user_clustering.py           # User segmentation and clustering
├── text_analyzer.py             # Advanced NLP analysis
├── utils.py                     # Utility functions
├── data/                        # Data directory
│   └── All_Beauty.json          # Amazon Beauty dataset
├── model/                       # Model directory
└── output/                      # Output directory for reports and visualizations
```

## Key Features

- **Advanced Data Processing**: Robust cleaning, feature extraction, handling of verified purchases
- **Sophisticated Feature Engineering**: User engagement, product popularity, sentiment analysis
- **User Clustering**: Personalized recommendations through user segmentation
- **Hyperparameter Tuning**: Cross-validation with parameter grid search
- **Multiple Evaluation Metrics**: RMSE, MAE, coverage, precision
- **Time-Based Analysis**: Temporal trends, seasonal patterns
- **Hybrid Recommendation**: Combination of collaborative filtering and content-based approaches
- **Real-time Recommendation**: Fast recommendation generation for production use

## Usage

### Training a Model

```bash
python run_recommendations.py --mode train --data_path ./data/All_Beauty.json
```

### Getting Recommendations

```bash
python run_recommendations.py --mode recommend --user_id A1CNRUZRD717UT --num_recs 10
```

### Evaluating Model Performance

```bash
python run_recommendations.py --mode evaluate
```

## Requirements

- PySpark 3.0+
- Python 3.7+
- pandas
- numpy
- matplotlib

## Implementation Details

### Data Processing

The system performs multiple preprocessing steps:
- Cleaning and filtering invalid reviews
- Computing recency features
- Weighting verified purchases
- Extracting features from review text

### Feature Engineering

Advanced features are created to improve recommendation quality:
- User engagement scores based on activity
- Product popularity metrics
- Review sentiment analysis
- Time-based features (recency, seasonal patterns)

### Model Training

The ALS (Alternating Least Squares) algorithm is used with:
- Hyperparameter tuning via cross-validation
- Regularization to prevent overfitting
- Cold-start handling strategies

### Evaluation Methodology

Multiple metrics are used to evaluate model performance:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Coverage (percentage of user-item pairs that can be predicted)
- Precision@k (accuracy of top-k recommendations)

### User Segmentation

K-means clustering is used to segment users based on:
- Rating behavior
- Engagement level
- Review sentiment
- Activity frequency

This enables more personalized recommendations tailored to user segments.

## Future Improvements

- Integration with product metadata for improved content-based filtering
- Implementation of deep learning recommendation models
- A/B testing framework for recommendation strategies
- Real-time feature updates for dynamic recommendations
