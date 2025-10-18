# kaggle_projects

## Forbes Billionaires 2024 — Data Analysis & ML Prediction

- Loaded and explored the 2024 Forbes Billionaires dataset (2,781 records).
- Cleaned data — handled missing values and converted “Net Worth” to numeric format.
- Performed EDA (Exploratory Data Analysis) using pandas, seaborn, and matplotlib to visualize industries, gender, age, residence, and wealth trends.
- Built a simple Random Forest Regression model to predict billionaire net worth based on age.
- Evaluated the model (RMSE ≈ 5.97, R² ≈ -0.20) and made a sample prediction for a 50-year-old billionaire (~$6.1B).


## Video Game Sales — Data Analysis & ML Prediction

- Loaded and explored the Video Game Sales dataset (16,598 records).
- Cleaned data — handled missing values and formatted the Year column as integer.
- Performed EDA using pandas, seaborn, and matplotlib to visualize sales trends by region, genre, platform, publisher, and year.
- Built a Random Forest Regression model to predict global sales based on Year, NA_Sales, EU_Sales, JP_Sales, Other_Sales, and platform/genre features.
- Evaluated the model (R² ≈ 0.82, RMSE ≈ 0.86) showing strong predictive performance.


## Google Search Trends — Data Analysis & ML Prediction

- Loaded and combined multiple CSVs of top 20 Google search queries per country.
- Cleaned data — filled missing locations and search terms, converted dates to datetime.
- Performed EDA using pandas, seaborn, and matplotlib to visualize top search locations, top search terms, word cloud, and search counts by period.
- Built a Random Forest Classifier to predict search period (Morning/Evening) using location and top 10 search terms.
- Built a CatBoost Classifier using location and top 20 search terms — all features treated as categorical.
- Evaluated models — CatBoost achieved ~60% accuracy.
- Demonstrated sample predictions using both models.
