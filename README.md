# Advanced Fraud Detection System  
## Project Overview  
This project develops a robust fraud detection system leveraging advanced machine learning techniques to identify fraudulent transactions across two distinct datasets: e-commerce user activity (Fraud_Data.csv) and credit card transactions (creditcard.csv). The goal is to build, train, and evaluate predictive models, with a strong emphasis on interpretability to understand the key drivers of fraudulent behavior.

## Problem Statement  
Fraud poses a significant threat to financial institutions and online businesses, leading to substantial financial losses and erosion of customer trust. The inherent challenge in fraud detection lies in the highly imbalanced nature of the data, where fraudulent transactions are extremely rare compared to legitimate ones. This project addresses this imbalance and aims to build models that can accurately flag fraudulent activities while maintaining high precision and recall.
## Project Structure  
This repository is organized into three key phases, each represented by a dedicated Jupyter Notebook:

1. **eda.ipynb**: Focuses on Data Analysis and Preprocessing.  
2. **modeling.ipynb**: Handles Model Building and Training.  
3. **explainability.ipynb**: Delves into Model Explainability using SHAP.  
## Datasets Used  
The project utilizes three datasets:

- **Fraud_Data.csv**: Contains user activity data including signup and purchase times, purchase value, device information, source, browser, sex, age, IP address, and a binary class label (0 for legitimate, 1 for fraudulent).

- **IpAddress_to_Country.csv**: Provides IP address ranges mapped to countries, used for enriching Fraud_Data.csv with geographical information.

- **creditcard.csv**: Contains anonymized credit card transaction data with numerical features (V1-V28), Time, Amount, and a binary Class label (0 for legitimate, 1 for fraudulent).
## Methodologies and Key Steps  
### Task 1: Data Analysis and Preprocessing (eda.ipynb)  
This notebook performs comprehensive data preparation:

**Data Loading & Initial Inspection**: Datasets are loaded and their initial structures, data types, and missing values are inspected.

**Missing Value Handling**: While raw datasets showed minimal explicit NaNs, SimpleImputer is strategically integrated into preprocessing pipelines to handle NaNs introduced during feature engineering (e.g., time_diff for first entries).

**Data Cleaning**:  
- Duplicate rows are identified and removed from all datasets to ensure data integrity.  
- Data types are corrected, notably converting signup_time and purchase_time to datetime objects and ip_address to int64 for efficient lookup.

**Exploratory Data Analysis (EDA)**:  
- **Univariate Analysis**: Distributions of key numerical features (e.g., purchase_value, age, Amount) and categorical features (e.g., source, browser, sex, class) are analyzed using histograms and count plots.  
- **Bivariate Analysis**: Relationships between features and the target variable (class/Class) are explored using box plots and grouped bar plots. Correlation matrices are used for numerical features in creditcard.csv.

**Geolocation Merging**:  
- IP addresses in Fraud_Data.csv are converted to integer format.  
- A custom function leveraging binary search (searchsorted) is implemented to efficiently map IP addresses to countries using IpAddress_to_Country.csv, enriching the Fraud_Data with a new country feature.

**Feature Engineering**: New informative features are derived from existing ones:  
- **Time-Based Features**: hour_of_day, day_of_week, day_of_year, month_of_year, week_of_year are extracted from purchase_time.  
- **time_since_signup**: Calculates the duration in seconds between user signup and purchase, providing insight into user behavior speed.  
- **Transaction Frequency and Velocity**: time_diff_user and time_diff_device capture the time elapsed since the last transaction for a given user or device, while device_transaction_count and user_transaction_count quantify activity levels.

**Data Transformation**:  
- **Class Imbalance Analysis**: The severe class imbalance in both fraud datasets is analyzed and acknowledged.  
- **Preprocessing Pipelines**: ColumnTransformer pipelines are defined for both datasets, incorporating SimpleImputer (for engineered NaNs), StandardScaler for numerical feature normalization, and OneHotEncoder for categorical feature encoding. These pipelines are fitted on training data and applied consistently.

### Task 2: Model Building and Training (modeling.ipynb)  
This notebook focuses on developing and evaluating machine learning models:

**Data Preparation**: Features (X) and target (y) variables are separated. A stratified train-test split (80% train, 20% test) is performed to maintain the original class distribution in both subsets, crucial for imbalanced data.

**Model Selection**: Two models are chosen for comparison:  
- **Logistic Regression**: Serves as a simple, interpretable baseline model.  
- **LightGBM (Gradient Boosting)**: A powerful ensemble model known for its speed and accuracy, particularly effective on tabular data.

**Handling Imbalance**: The SMOTE (Synthetic Minority Over-sampling Technique) algorithm is integrated into imblearn.pipeline.Pipeline for both models. SMOTE is applied only to the training data to generate synthetic samples of the minority class, effectively balancing the dataset without data leakage from the test set. This approach helps models learn robust fraud patterns.

**Model Training**: Both Logistic Regression and LightGBM models are trained within their respective pipelines on the preprocessed and SMOTE-augmented training data.

**Evaluation Metrics**: Given the imbalanced nature of the datasets, appropriate metrics are used to assess model performance:  
- **F1-Score**: Harmonic mean of precision and recall, balancing false positives and false negatives.  
- **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**: Measures the model's ability to distinguish between classes across all possible classification thresholds.  
- **AUC-PR (Area Under the Precision-Recall Curve)**: Particularly informative for highly imbalanced datasets, focusing on the trade-off between precision and recall for the positive class.  
- **Confusion Matrix**: Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

**Key Findings/Best Model**:  
Based on the provided output, LightGBM consistently outperforms Logistic Regression on both datasets, showing significantly higher F1-Scores and AUC-PR, indicating its superior ability to detect fraud effectively with fewer false alarms. For example, on CreditCard_Data, LightGBM achieved an F1-Score of 0.8488 and AUC-PR of 0.8038, far surpassing Logistic Regression's 0.0987 and 0.7164 respectively.

### Task 3: Model Explainability (explainability.ipynb)  
This notebook focuses on interpreting the best-performing models using SHAP:

**Methodology**: SHAP (Shapley Additive exPlanations) is employed to explain the output of the LightGBM models. SHAP values attribute the contribution of each feature to a prediction, providing both global and local interpretability.

**Global Feature Importance (SHAP Summary Plots)**:  
- SHAP Summary Plots (bar charts) are generated for both Fraud_Data and CreditCard_Data.  
- These plots visually represent the overall impact of each feature on the model's output. Features are ranked by their average absolute SHAP value, showing which features are most important across the entire dataset. The direction of impact (positive or negative contribution to fraud prediction) is also visible in the dot plot version (though bar plot is used for reliability here).

**Local Feature Importance (SHAP Force Plots)**:  
- SHAP Force Plots are generated for individual fraudulent and non-fraudulent transactions.  
- These plots illustrate how each feature pushes the model's output from the base value (average prediction) to the final prediction for a specific instance. Red indicates features increasing the prediction (e.g., towards fraud), while blue indicates features decreasing it.

**Key Insights from SHAP**:
- The SHAP plots will reveal the most influential features in predicting fraud for both datasets. For instance, in CreditCard_Data, features like V14, V4, and V12 are likely to show high importance, indicating their strong role in distinguishing fraudulent transactions.
- For Fraud_Data, features related to time_since_signup, purchase_value, country, and device_transaction_count are expected to be significant. The force plots will provide specific examples of how these features combine to drive individual fraud predictions.

## Technical Stack  
- **Python**: Programming Language  
- **Pandas**: Data manipulation and analysis  
- **NumPy**: Numerical operations  
- **Scikit-learn**: Machine learning models, preprocessing, and evaluation  
- **Imbalanced-learn (imblearn)**: Handling imbalanced datasets (e.g., SMOTE)  
- **LightGBM**: Gradient Boosting Machine learning framework  
- **Matplotlib**: Plotting and visualization  
- **Seaborn**: Enhanced statistical data visualization  
- **SHAP**: Model interpretability (Shapley Additive exPlanations)
## How to Run the Project

To run this project, please follow the steps below:

### 1. Prerequisites
- Ensure you have **Python 3.8+** installed.
- Install **Jupyter Notebook** or **JupyterLab** to run the `.ipynb` files.

### 2. Clone the Repository
```bash
git clone <https://github.com/Shegaw-21hub/FraudDetection>
cd <FraudDetection>
```
### 3. Set Up Virtual Environment (Recommended)
```
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
### 4. Install Dependencies
```
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm imbalanced-learn shap
```
### 5. Place Data Files  
Download the following datasets:
- Fraud_Data.csv  
- IpAddress_to_Country.csv  
- creditcard.csv

```
your_project/
├── notebooks/
│   ├── eda.ipynb
│   ├── modeling.ipynb
│   └── explainability.ipynb
└── data/
    ├── Fraud_Data.csv
    ├── IpAddress_to_Country.csv
    └── creditcard.csv 
```
### 6.Run Jupyter Notebook/Lab:
```
jupyter notebook
# or
jupyter lab
```
### 7. Execute Notebooks:

- Navigate to the `notebooks/` directory.
- Open `eda.ipynb` and run all cells (Cell -> Run All).
- Open `modeling.ipynb` and run all cells (Cell -> Run All).
- Open `explainability.ipynb` and run all cells (Cell -> Run All).

**Important:** If you encounter any issues, especially with plot rendering, perform a Kernel -> Restart Kernel and Clear All Outputs before running all cells again.
## Results and Performance Summary  
The models were evaluated using F1-Score, AUC-ROC, and AUC-PR, which are critical for imbalanced datasets.
| Dataset         | Model              | F1-Score | AUC-ROC | AUC-PR |
|-----------------|--------------------|----------|---------|--------|
| Fraud_Data      | Logistic Regression | 0.6273   | 0.8406  | 0.6566 |
| Fraud_Data      | LightGBM           | 0.6910   | 0.8445  | 0.7141 |
| CreditCard_Data | Logistic Regression | 0.0987   | 0.9625  | 0.7164 |
| CreditCard_Data | LightGBM           | 0.8488   | 0.9723  | 0.8038 |

- LightGBM consistently demonstrated superior performance across both datasets, particularly excelling in F1-Score and AUC-PR, making it the chosen model for explainability.
- Its ability to handle complex relationships and high-dimensional data, combined with the SMOTE oversampling, proved highly effective in detecting fraudulent transactions.
## Conclusion  
This project successfully developed and evaluated a machine learning-based fraud detection system, demonstrating proficiency in data preprocessing,



