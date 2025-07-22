from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

def train_logistic_regression_pipeline(preprocessor, X_train, y_train, sampling_strategy=1.0):
    """
    Trains a Logistic Regression model within an imblearn pipeline.
    """
    pipeline_lr = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=sampling_strategy)),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'))
    ])
    print("Training Logistic Regression...")
    pipeline_lr.fit(X_train, y_train)
    print("Logistic Regression training complete.")
    return pipeline_lr

def train_lightgbm_pipeline(preprocessor, X_train, y_train, sampling_strategy=1.0):
    """
    Trains a LightGBM model within an imblearn pipeline.
    """
    pipeline_lgbm = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=sampling_strategy)),
        ('classifier', LGBMClassifier(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31))
    ])
    print("Training LightGBM...")
    pipeline_lgbm.fit(X_train, y_train)
    print("LightGBM training complete.")
    return pipeline_lgbm
