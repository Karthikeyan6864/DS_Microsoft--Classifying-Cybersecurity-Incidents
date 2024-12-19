import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'new_train_sample.csv'  # Replace with your CSV file name
df = pd.read_csv(file_path)

# Print the first few rows of the DataFrame
print(df.head())

# Describe the DataFrame (statistical summary)
print(df.describe())

print(df.shape)
from sklearn.model_selection import train_test_split

# Drop rows with NaN values in 'IncidentGrade'
df_cleaned = df.dropna(subset=['IncidentGrade'])
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("Percentage of missing values in each column:")
print(missing_percentage)
# Perform stratified sampling
#sample_df, _ = train_test_split(df_cleaned, test_size=0.9, stratify=df_cleaned['IncidentGrade'], random_state=42)

# Output the result

#print("Rows before cleaning:", df.shape[0])
#print("Rows after cleaning:", df_cleaned.shape[0])
#print("Sampled data shape:", sample_df.shape)

#missing_values = sample_df.isnull().sum()
#print(missing_values)

#for column in sample_df.columns:
#    print(f"Unique values in {column}:")
#    print(sample_df[column].value_counts())
#    print("\n")

#undersample the majority class
#import pandas as pd
#from sklearn.utils import resample

# Assuming 'data' is your DataFrame and 'IncidentGrade' is the target column
# Split data into majority and minority classes
#majority_class = sample_df[sample_df['IncidentGrade'] == 'BenignPositive']
#class_2 = sample_df[sample_df['IncidentGrade'] == 'TruePositive']
#minority_class = sample_df[sample_df['IncidentGrade'] == 'FalsePositive']

# Undersample the majority class to match the minority class count
#majority_undersampled = resample(
#    majority_class,
#    replace=False,  # Sampling without replacement
#    n_samples=len(minority_class),  # Match the minority class size
#    random_state=42  # Ensure reproducibility
#)

# Combine the undersampled majority class with the other classes
#balanced_data = pd.concat([majority_undersampled, class_2, minority_class])

# Shuffle the dataset
#balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Check new class distribution
#print(balanced_data['IncidentGrade'].value_counts())
#balanced_data.describe()
#balanced_data.info()
#balanced_data.to_csv('balanced_data.csv', index=False)