# COVID-19-Clinical-Trials--Analysis-and-Prediction
This project analyzes COVID clinical trials using Python. It includes data cleaning, feature engineering (e.g., trial duration, country extraction), exploratory analysis (visualizations, correlations), and predictive modeling using Random Forest. Outputs include feature importance, classification reports, and insights.
# COVID Clinical Trials Analysis

This project explores and analyzes a dataset of COVID-19 clinical trials. It covers data cleaning, feature engineering, exploratory data analysis (EDA), visualizations, and predictive modeling using machine learning techniques.

## Project Features

- **Data Cleaning**: Handles missing data using imputation techniques (mode for categorical and median for numerical data). Drops irrelevant columns.
- **Feature Engineering**: 
  - Extracts `Country` from the `Locations` column.
  - Calculates trial duration using `Start Date` and `Completion Date`.
- **Exploratory Data Analysis (EDA)**: 
  - Visualizes key data distributions, including trial status, phases, and enrollment.
  - Computes and displays a correlation matrix for numerical features.
- **Predictive Modeling**: 
  - Implements a binary classification model to predict trial status (`Completed` vs. others) using a Random Forest Classifier.
  - Evaluates the model with metrics like classification reports, confusion matrices, and feature importance.

## Dataset

The dataset includes information about COVID-19 clinical trials. Columns such as `Rank`, `Title`, `Status`, `Conditions`, `Enrollment`, and `Phases` are analyzed.

## Requirements

Install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Project Structure

- **`cTrial.py`**: Main script for the project.
- **`COVID_clinical_trials.csv`**: Dataset file (update file path in the script).
- **Output**: Visualizations (e.g., trial status distribution, correlation matrix), feature importance, and model evaluation metrics.

## Running the Project

1. Clone the repository.
2. Place the `COVID_clinical_trials.csv` file in the project directory.
3. Update the file path in the script.
4. Run the Python script:
   ```bash
   python cTrial.py
   ```

## Outputs

- **Data Visualizations**: Trial status and phase distributions, enrollment histogram, and heatmap of correlations.
- **Model Evaluation**:
  - Classification Report
  - Confusion Matrix
  - Feature Importance Plot
