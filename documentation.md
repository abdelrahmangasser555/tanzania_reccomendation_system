<VSCode.Cell language="markdown">

# Tanzania Tourism Cost Prediction Project

This project analyzes tourism data from Tanzania to predict tourist expenditure costs using machine learning techniques.

## Project Overview

The goal is to build a predictive model that can estimate how much tourists will spend during their visit to Tanzania based on various factors like demographics, travel patterns, and accommodation preferences.

## Dataset Description

### Training Data

- **Size**: ~4000 samples
- **Target Variable**: `total_cost` (tourist expenditure in USD)
- **Features**: Demographics, travel patterns, accommodation details, activities

### Key Features

- `country`: Tourist's country of origin
- `age_group`: Age categories (1-24, 25-44, 45-64, 65+)
- `travel_with`: Travel companions (alone, family, friends, etc.)
- `total_female`, `total_male`: Number of tourists by gender
- `night_mainland`, `night_zanzibar`: Nights spent in different regions
- `purpose`: Travel purpose (leisure, business, etc.)
- `main_activity`: Primary tourist activity
- `package_*`: Various package inclusions (transport, accommodation, etc.)
- `payment_mode`: Payment method used

## Methodology

### 1. Data Cleaning and Preprocessing

- Handle missing values using mode for categorical and median for numerical
- Remove duplicate records
- Standardize text fields to lowercase
- Drop rows with missing target values

### 2. Feature Engineering

- **Derived Features**:
  - `total_tourists` = total_female + total_male
  - `total_nights` = night_mainland + night_zanzibar
  - `cost_per_night` = total_cost / total_nights
  - `cost_per_person` = total_cost / total_tourists
- **Binary Encoding**: Convert package services to binary features
- **Categorical Mapping**: Map age groups and purposes to numerical values
- **One-Hot Encoding**: For categorical variables like country and activities

### 3. Machine Learning Models

#### Supervised Learning Models Tested:

1. **Linear Regression**: Baseline model
2. **Ridge Regression**: L2 regularization to prevent overfitting
3. **Lasso Regression**: L1 regularization for feature selection
4. **Random Forest**: Ensemble method with decision trees
5. **Gradient Boosting**: Sequential ensemble learning

#### Model Evaluation Metrics:

- **RMSE** (Root Mean Square Error): Measures prediction accuracy
- **R² Score**: Explains variance in the data
- **MAE** (Mean Absolute Error): Average prediction error

### 4. Unsupervised Learning

- **Customer Segmentation**: K-means clustering to identify tourist segments
- **PCA Visualization**: 2D representation of customer clusters
- **Elbow Method**: Determine optimal number of clusters

### 5. Recommendation System

- **Similarity-based**: Find similar tourists using cosine similarity
- **Activity Recommendations**: Suggest popular activities by cluster
- **Country Preferences**: Identify destination patterns

## Results Summary

### Model Performance

The best performing model typically achieves:

- **R² Score**: 0.4-0.7 (explains 40-70% of cost variance)
- **RMSE**: $2000-5000 (prediction error range)

### Key Insights

1. **Country of Origin** strongly influences spending patterns
2. **Length of Stay** (total nights) correlates with total cost
3. **Package Services** affect expenditure significantly
4. **Age Groups** show different spending behaviors
5. **Business Travelers** tend to spend more than leisure tourists

### Customer Segments Identified

- **Budget Backpackers**: Young, short stays, minimal packages
- **Family Vacationers**: Medium spending, longer stays
- **Luxury Tourists**: High spending, full service packages
- **Business Travelers**: Consistent high spending
- **Adventure Seekers**: Activity-focused spending

## Technical Implementation

### Libraries Used

```python
# Data Processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
```

### Project Structure

```
data analytics/
├── Train.csv                    # Training dataset
├── Test.csv                     # Test dataset for predictions
├── SampleSubmission.csv         # Submission format
├── data_cleaning_v1.ipynb       # Main analysis notebook
├── final_tourism_predictions.csv # Model predictions
└── documentation.md             # This documentation
```

## Usage Instructions

### Running the Analysis

1. **Load Data**: Execute cell 1 to import libraries and load datasets
2. **Data Exploration**: Run cells 2-5 for initial data analysis
3. **Feature Engineering**: Execute cells 6-7 to create new features
4. **Model Training**: Run cells 8-11 to train all models
5. **Generate Predictions**: Execute final cells for test predictions

### Key Functions

- `run_complete_analysis()`: Complete data cleaning and visualization
- `run_complete_ml_analysis()`: Train all ML models
- `run_competition_predictions()`: Generate final predictions
- `interactive_prediction_demo()`: Test model with sample scenarios

## Business Applications

### Tourism Industry

- **Pricing Strategy**: Set competitive tour packages
- **Market Segmentation**: Target different tourist types
- **Revenue Forecasting**: Predict seasonal income
- **Resource Allocation**: Plan for high-spending visitors

### Government Planning

- **Infrastructure Development**: Focus on high-value tourism areas
- **Marketing Budget**: Allocate resources to profitable markets
- **Policy Making**: Support tourism growth strategies

## Future Improvements

### Model Enhancement

- **Advanced Algorithms**: Try XGBoost, Neural Networks
- **Feature Engineering**: Add seasonal, economic indicators
- **Ensemble Methods**: Combine multiple model predictions
- **Cross-Validation**: Better model validation techniques

### Data Collection

- **More Features**: Weather, events, exchange rates
- **Larger Dataset**: More samples for better training
- **Real-time Data**: Dynamic pricing based on current trends
- **External Data**: Economic indicators, competitor prices

## Conclusion

This project successfully demonstrates how machine learning can predict tourism costs with reasonable accuracy. The insights gained help understand tourist behavior patterns and can inform business decisions in the tourism industry.

The combination of supervised learning for prediction and unsupervised learning for segmentation provides a comprehensive analysis framework that can be adapted for other tourism markets or extended with additional data sources.
</VSCode.Cell>
