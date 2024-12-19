import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('GUIDE_Test.csv')

missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Calculate the percentage of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of missing values in each column:")
print(missing_percentage)

# Drop columns with missing values exceeding the threshold
threshold = 0.5  # 50%
missing_percentage = df.isnull().mean()
df = df.loc[:, missing_percentage <= threshold]
print("\nDataFrame info after dropping columns with excessive missing values:")
print(df.info())




encoder = LabelEncoder()
df['IncidentGrade'] = pd.Categorical(df['IncidentGrade'], categories=['FalsePositive', 'BenignPositive', 'TruePositive'], ordered=True)
df['IncidentGrade_Encoded'] = encoder.fit_transform(df['IncidentGrade'])
df['EvidenceRole_Encoded'] = encoder.fit_transform(df['EvidenceRole'])
df['Category_Encoded'] = encoder.fit_transform(df['Category'])
df['EntityType_Encoded'] = encoder.fit_transform(df['EntityType'])
print(df)


feature_columns = ['EvidenceRole_Encoded', 'Category_Encoded', 'EntityType_Encoded','DetectorId','AlertTitle','OrgId']  # Replace with your actual features
target_column = 'IncidentGrade_Encoded'  # Replace with your target column



decision_tree_model = joblib.load('decision_tree_model.pkl')


random_forest_model = joblib.load('random_forest_model.pkl')



X_test = df[feature_columns]
y_test = df[target_column]

# Decision Tree Predictions and Evaluation
print("\nEvaluating Decision Tree Model...")
dt_predictions = decision_tree_model.predict(X_test)


dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")


dt_cm = confusion_matrix(y_test, dt_predictions)
print("\nDecision Tree Confusion Matrix:")
print(dt_cm)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_predictions))


# Random Forest Predictions and Evaluation
print("\nEvaluating Random Forest Model...")
rf_predictions = random_forest_model.predict(X_test)


rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")


rf_cm = confusion_matrix(y_test, rf_predictions)
print("\nRandom Forest Confusion Matrix:")
print(rf_cm)


plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# Compare Results
print(f"\nModel Comparison:")
print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
