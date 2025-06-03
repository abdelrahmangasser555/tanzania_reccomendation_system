# Tourism Cost Prediction Project Report

## 1. Introduction and Objective

The objective of this project is to develop a machine learning model that can predict tourism costs based on various factors such as accommodation preferences, location choices, and other travel-related features. The project involves building a regression model to estimate the total cost of tourism packages, which can help travel agencies and customers make informed decisions about pricing and planning.

The dataset contains information about different tourism packages including details about nights spent in Zanzibar, mainland destinations, and associated costs. Our goal is to create an accurate predictive model that can estimate total tourism costs based on these input features.

## 2. Data Cleaning Process

### Initial Data Exploration
- **Dataset Loading**: Loaded the training dataset from "Train.csv"
- **Basic Information Gathering**: 
  - Examined first 5 rows to understand data structure
  - Checked dataset information including data types and column names
  - Identified missing values across all columns
  - Detected duplicate entries in the dataset

### Missing Value Treatment
- **Numerical Columns**: Missing values were imputed using the median of each respective column
- **Categorical Columns**: Missing values were filled using the mode (most frequent value) of each column
- This approach ensures that the imputation doesn't significantly skew the distribution of the data

### Outlier Detection and Treatment
- **Outlier Identification**: Used the Interquartile Range (IQR) method to detect outliers
  - Calculated Q1 (25th percentile) and Q3 (75th percentile)
  - Defined outliers as values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
- **Outlier Treatment**: Applied capping method where outliers were replaced with the boundary values (lower_bound and upper_bound)

## 3. Data Visualization

### Distribution Analysis
- **Kernel Density Estimation (KDE) Plots**: Created KDE plots for all numerical columns to understand the distribution shape and identify skewness
- **Box Plots**: Generated box plots to visualize the spread of data and confirm outlier removal effectiveness

### Correlation Analysis
- **Heatmap Visualization**: Created correlation heatmaps to understand relationships between variables
- **Boolean to Numerical Conversion**: Converted boolean variables (night_zanzibar, night_mainland) to numerical format for correlation analysis
- The heatmap revealed correlations between location preferences and total costs

### Before and After Comparisons
- **Normalization Comparison**: Visualized the effect of Min-Max normalization using histogram overlays
- **Outlier Treatment Verification**: Compared box plots before and after outlier treatment to confirm effectiveness

## 4. Feature Engineering

### Data Transformation
- **Target Variable Transformation**: Applied logarithmic transformation to the 'total_cost' variable to reduce skewness and achieve a more normal distribution
- **Skewness Analysis**: Calculated skewness before and after transformation to measure improvement

### Scaling and Normalization
- **Feature-Specific Scaling**: 
  - Applied Min-Max scaling to features with high skewness (< -0.5 or > 0.5)
  - Applied Standard Scaling (Z-score) to features with moderate skewness (-0.5 to 0.5)
- **Rationale**: Different scaling methods were chosen based on the distribution characteristics of each feature

### Categorical Encoding
- **Label Encoding**: Applied label encoding to categorical variables to convert them into numerical format
- **One-Hot Encoding**: Also experimented with one-hot encoding using pandas get_dummies for categorical features
- **Boolean Handling**: Converted boolean variables to binary (0/1) format for model compatibility

## 5. Model Development and Training

### Data Preparation
- **Data Filtering**: Removed rows with zero or negative total_cost values to avoid issues with log transformation
- **Train-Test Split**: Split the data into 80% training and 20% testing sets
- **Feature Scaling**: Applied StandardScaler to numerical features only

### Model Selection
- **Gradient Boosting Regressor**: Chosen as the primary model due to its effectiveness with mixed data types and robustness to outliers
- **Model Parameters**:
  - n_estimators: 200
  - learning_rate: 0.1
  - max_depth: 4
  - random_state: 42 (for reproducibility)

### Model Evaluation
- **Performance Metrics**:
  - RMSE (Root Mean Square Error) on both log and original scales
  - R² Score to measure the proportion of variance explained
- **Prediction Generation**: Created predictions for test dataset and prepared submission file

## 6. Key Technical Implementations

### Data Quality Assurance
- Comprehensive duplicate detection and removal
- Systematic handling of infinite values and NaN entries
- Validation of data types and column consistency

### Statistical Analysis
- Skewness and kurtosis calculations for distribution analysis
- Quantile-based outlier detection methodology
- Correlation matrix generation for feature relationship analysis

### Model Validation
- Cross-validation approach using train-test split
- Performance evaluation on both transformed and original scales
- Prediction export functionality for real-world application

## 7. Conclusion

This project successfully developed a comprehensive tourism cost prediction system through a systematic data science approach. The key achievements include:

**Data Quality**: Implemented robust data cleaning procedures that handled missing values, outliers, and data type inconsistencies effectively.

**Feature Engineering**: Applied appropriate transformations including logarithmic scaling for the target variable and feature-specific normalization techniques that improved model performance.

**Modeling Success**: The Gradient Boosting Regressor demonstrated strong predictive capabilities, with the model able to explain a significant portion of the variance in tourism costs.

**Practical Application**: The final model generates predictions that can be directly used by travel agencies and tourism platforms for pricing strategies and cost estimation.

**Technical Rigor**: The project followed best practices in data science including proper train-test splitting, appropriate scaling techniques, and comprehensive evaluation metrics.

The systematic approach to data preprocessing, combined with careful feature engineering and model selection, resulted in a robust predictive system that can provide valuable insights for tourism cost estimation. The methodology developed here can be extended to other domains requiring cost prediction with mixed data types and complex feature relationships.

Future improvements could include ensemble methods, hyperparameter tuning, and incorporation of additional external factors such as seasonal variations and economic indicators.
