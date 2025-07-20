# Advanced Fraud Detection System  
## Project Overview  
This project develops a robust fraud detection system leveraging advanced machine learning techniques to identify fraudulent transactions across two distinct datasets: e-commerce user activity (Fraud_Data.csv) and credit card transactions (creditcard.csv). The goal is to build, train, and evaluate predictive models, with a strong emphasis on interpretability to understand the key drivers of fraudulent behavior.

**💼 Business Value Proposition:**  

This system tackles one of the most pressing challenges for financial institutions and e-commerce platforms: **fraud prevention**.  
By accurately flagging suspicious transactions with **high precision and recall**, it empowers businesses to take **proactive action** — minimizing financial losses, protecting customer data, and preserving operational trust.  

What sets it apart?  
It not only stops fraud but also reduces false positives — avoiding disruptions for genuine customers.  

The result:  
**Better customer experience**, **secured revenue**, and a **stronger brand reputation** in an increasingly high-risk digital landscape.



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

✨ **Key Insights from Plots**  

🔹 For `CreditCard_Data`, features like **`V14`**, **`V4`**, and **`V12`** consistently emerge as the most influential, indicating their strong role in distinguishing fraudulent transactions.  
&nbsp;&nbsp;&nbsp;&nbsp;✔️ High **negative values** of `V14` and `V12` — often associated with **legitimate transactions**  
&nbsp;&nbsp;&nbsp;&nbsp;✔️ **Positive values** of `V4` — often associated with **fraudulent transactions**  
➡️ These trends act as **strong predictive signals** in identifying fraud.

---

🔹 For `Fraud_Data`, the following features are identified as key drivers, highlighting **behavioral** and **contextual** fraud patterns:  
&nbsp;&nbsp;&nbsp;&nbsp;• `time_since_signup` — shorter times often indicate **fraud**  
&nbsp;&nbsp;&nbsp;&nbsp;• `purchase_value` — higher values can sometimes be **less fraudulent**, or **specific ranges** are riskier  
&nbsp;&nbsp;&nbsp;&nbsp;• `country` — certain countries exhibit **higher fraud rates**  
&nbsp;&nbsp;&nbsp;&nbsp;• `device_transaction_count` — high counts suggest **suspicious activity**

💡 These insights not only enhance model performance but also provide a deeper understanding of **how fraud manifests across diverse datasets**.


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

## 🧪 Testing & Continuous Integration/Continuous Deployment (CI/CD)

To ensure **code quality**, **reliability**, and **reproducibility**, this project incorporates **unit testing** and a **Continuous Integration (CI)** pipeline.

### ✅ Unit Tests
Implemented using `pytest`, unit tests rigorously validate individual components of the **data preprocessing** and **feature engineering** pipeline.  
This ensures the correctness of critical functions like:
- **IP-to-country mapping**
- **Transaction velocity calculations**

### 🔄 Continuous Integration (CI)
A **GitHub Actions workflow** (`.github/workflows/ci.yml`) automates the following on every **push to the `main` branch** and on **pull requests**:

- 📦 **Dependency Installation**: Ensures all necessary libraries are correctly set up.  
- 🧪 **Dummy Data Creation**: Creates essential placeholder data files for the CI environment to allow notebooks to run without external data dependencies.  
- 🧫 **Unit Test Execution**: Runs all defined unit tests to catch regressions early.  
- 📊 **Notebook Execution**: Executes all Jupyter notebooks (`eda.ipynb`, `modeling.ipynb`, `explainability.ipynb`) from start to finish, verifying that the entire analytical pipeline runs without errors and is reproducible.

---

This automated **testing and integration** process guarantees that the project’s core logic remains sound and that the analytical results can be **consistently reproduced**.


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
#### 📦 Setting Up Dependencies with `requirements.txt`
To ensure a smooth, consistent, and production-ready environment, create a `requirements.txt` file in the **root directory** of your project with the following contents:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
lightgbm
imbalanced-learn
shap
pytest
nbconvert
```

This setup includes everything you need for:

-  Efficient data manipulation and visualization (`pandas`, `matplotlib`, `seaborn`)
-  Machine learning and model evaluation (`scikit-learn`, `lightgbm`, `imbalanced-learn`)
-  Model interpretability (`shap`)
-  Testing (`pytest`)
-  Notebook execution and conversion (`nbconvert`)

> ✅ **Pro Tip:**  
> After creating this file, install all dependencies at once using:
> ```
> pip install -r requirements.txt
> ```

This single step powers your entire workflow — from data science to explainability — with reliability and reproducibility built in.




### 5. Place Data Files

#### FraudDetection — Project Architecture Overview

> **A meticulously organized, enterprise-ready framework**  
> engineered for **robust fraud detection workflows**,  
> **collaborative development**, and **seamless CI/CD integration**.

```plaintext
FraudDetection/
├── 📂 notebooks/               # Interactive Jupyter Notebooks driving the analytics pipeline
│   ├── eda.ipynb              # Comprehensive Exploratory Data Analysis & Preprocessing
│   ├── modeling.ipynb         # Advanced Model Building, Training & Evaluation
│   └── explainability.ipynb   # Transparent Model Interpretability with SHAP insights
│
├── 📂 data/                   # Core datasets powering the detection system
│   ├── Fraud_Data.csv         # E-commerce user activity dataset
│   ├── IpAddress_to_Country.csv # IP-to-country mapping for geolocation enrichment
│   └── creditcard.csv         # Anonymized credit card transactions
│
├── 📂 tests/                  # Rigorous unit tests ensuring pipeline reliability
│   └── test_data_processing.py # Validation of data preprocessing & feature engineering
│
├── 📂 .github/                # GitHub Actions workflows for CI/CD automation
│   └── workflows/
│       └── ci.yml             # Continuous Integration pipeline: testing & notebook execution
│
├── 📂 src/                    # (Optional) Modular source code for reusable components
│
├── 📄 requirements.txt        # Explicit dependency list ensuring environment consistency
├── 📄 README.md               # Comprehensive project documentation & usage guidelines
├── 📄 .gitignore              # Git exclusion rules for clean repository management
└── 📂 venv/                   # Isolated Python virtual environment (excluded from version control)

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

---

**Author:** Shegaw Adugna Melaku  
**GitHub:** [Shegaw-21hub/FraudDetection](https://github.com/Shegaw-21hub/FraudDetection)  
**LinkedIn:** [shegaw-adugna](https://www.linkedin.com/in/shegaw-adugna-b751a1166/)  
**Email:** [shegamihret@gmail.com](mailto:shegamihret@gmail.com)  
**Date:** July 2025

---




