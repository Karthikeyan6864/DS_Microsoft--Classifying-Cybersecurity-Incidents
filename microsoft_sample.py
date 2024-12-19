import pandas as pd
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import randint
import joblib


df = pd.read_csv('balanced_data.csv')


missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)


missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of missing values in each column:")
print(missing_percentage)

# Drop columns with missing values exceeding the threshold
threshold = 0.5  # 50%
missing_percentage = df.isnull().mean()
df = df.loc[:, missing_percentage <= threshold]
print("\nDataFrame info after dropping columns with excessive missing values:")
print(df.info())

# outlier detection
selected_columns = ['DetectorId', 'AlertTitle', 'Category', 'EntityType', 'EvidenceRole']  # Replace with column names


numeric_columns = [col for col in selected_columns if pd.api.types.is_numeric_dtype(df[col])]

if numeric_columns:
    z_scores = df[numeric_columns].apply(zscore, nan_policy='omit')
    outliers = (z_scores > 3) | (z_scores < -3)
    print("\nOutliers in selected numeric columns:")
    print(df[outliers.any(axis=1)])
else:
    print("\nNo numeric columns available for outlier detection.")

# Distribution analyze
common_column = 'IncidentGrade'


if common_column in df.columns:
    for col in selected_columns:
        if col in df.columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df, x=col, hue=common_column, kde=True, palette='Set2', bins=30)
            plt.title(f'Distribution of {col} by {common_column}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.legend(title=common_column, loc='upper right')
            plt.show()
else:
    print(f"\nColumn '{common_column}' is not present in the DataFrame.")

#Label Encoding
encoder = LabelEncoder()
df['IncidentGrade'] = pd.Categorical(df['IncidentGrade'], categories=['FalsePositive', 'BenignPositive', 'TruePositive'], ordered=True)

df['IncidentGrade_Encoded'] = encoder.fit_transform(df['IncidentGrade'])
df['EvidenceRole_Encoded'] = encoder.fit_transform(df['EvidenceRole'])
df['Category_Encoded'] = encoder.fit_transform(df['Category'])
df['EntityType_Encoded'] = encoder.fit_transform(df['EntityType'])
print(df)


feature_columns = ['EvidenceRole_Encoded', 'Category_Encoded', 'EntityType_Encoded','DetectorId','AlertTitle','OrgId']  # Replace with your actual features
target_column = 'IncidentGrade_Encoded' 



X = df[feature_columns]
y = df[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)




# Create a function to evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Testing Accuracy: {test_accuracy:.2f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

# Decision Tree Hyperparameter Tuning
dt_param_dist = {
    'classifier__max_depth': [3, 5, 10, None],
    'classifier__min_samples_split': randint(2, 11),
    'classifier__min_samples_leaf': randint(1, 5),
}

dt_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42))
     ])

dt_random = RandomizedSearchCV(
    estimator=dt_pipeline,
    param_distributions=dt_param_dist,
    n_iter=20,  
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

print("Tuning Decision Tree with RandomizedSearchCV...")
dt_random.fit(X_train, y_train)
print(f"Best Parameters for Decision Tree: {dt_random.best_params_}")
print(f"Best Cross-Validation Accuracy for Decision Tree: {dt_random.best_score_:.2f}")

# Evaluate the tuned Decision Tree model
print("\nDecision Tree Model Evaluation:")
evaluate_model(dt_random.best_estimator_, X_train, X_test, y_train, y_test)


# Random Forest Hyperparameter Tuning
rf_param_dist = {
    'classifier__n_estimators': randint(50, 201),
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': randint(2, 11),
    'classifier__min_samples_leaf': randint(1, 5),
    'classifier__max_features': ['sqrt', 'log2']
}

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_random = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_param_dist,
    n_iter=20,  
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

print("Tuning Random Forest with RandomizedSearchCV...")
rf_random.fit(X_train, y_train)
print(f"Best Parameters for Random Forest: {rf_random.best_params_}")
print(f"Best Cross-Validation Accuracy for Random Forest: {rf_random.best_score_:.2f}")

# Evaluate the tuned Random Forest model
print("\nRandom Forest Model Evaluation:")
evaluate_model(rf_random.best_estimator_, X_train, X_test, y_train, y_test)


# Save the tuned Decision Tree model
joblib.dump(dt_random.best_estimator_, 'decision_tree_model.pkl')

# Save the tuned Random Forest model
joblib.dump(rf_random.best_estimator_, 'random_forest_model.pkl')

