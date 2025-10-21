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


## AI-Powered Resume Screening — Job Fit Prediction

- Loaded and explored a resume dataset with 1,000 records.
- Cleaned missing certifications and standardized text fields like Skills, Education, and Job Role.
- Visualized experience, salary, education, and AI score trends using charts.
- Built a Random Forest model to predict if a candidate would be hired.
- Tested my resume across all roles to see which roles have the highest chance of hire.
- Output shows the best-fit job roles for my skills and experience.


## TED Talks Analysis & Virality Prediction

- Loaded and explored a TED Talks dataset with 4,641 records.
- Cleaned missing numeric and text fields, converted durations to seconds, and extracted release date features.
- Performed EDA to analyze top speakers, popular categories, view distributions, and engagement metrics.
- Built an XGBoost model to predict viral talks (top 10% by views), achieving 98% accuracy.
- Implemented a content-based recommendation system using TF-IDF and cosine similarity to suggest similar talks.
- Tools used: Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost.


## Animal Face Classification

- Loaded and explored the AFHQ dataset with cat, dog, and wild images.  
- Applied image preprocessing and augmentation for training and validation.  
- Built a ResNet18 model using transfer learning to classify animal faces.  
- Trained the model for 5 epochs, achieving high accuracy on validation data.  
- Implemented prediction for new images, displaying the animal class.  
- Tools used: Python, PyTorch, torchvision, PIL, NumPy.






