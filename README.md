
# Ex. No. 3b – Data Preprocessing

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Preprocessing-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Experiment-Completed-success)

## AIM

To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset into training and testing sets, and applying feature scaling.

## OBJECTIVES

- Load and inspect a dataset using Pandas.
- Separate independent variables and the dependent variable.
- Handle missing numerical values using `SimpleImputer`.
- Encode categorical input features using One-Hot Encoding.
- Encode the target variable using `LabelEncoder`.
- Divide the dataset into training and testing sets.
- Apply feature scaling to numerical features.
- Prepare the dataset for machine-learning model training.

## SOFTWARE REQUIREMENTS

| Component | Description |
|---|---|
| Programming language | Python 3.x |
| Data handling | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Environment | Google Colab or Jupyter Notebook |
| Input dataset | `Data.csv` |

## THEORY

### 1. Data Preprocessing

Data preprocessing is the process of converting raw data into a clean and suitable format for machine-learning algorithms.

Real-world datasets may contain missing values, categorical attributes, inconsistent formats, and numerical features with different scales. These issues can reduce the accuracy and reliability of machine-learning models.

The main preprocessing operations performed in this experiment are:

- Missing-value handling
- Categorical data encoding
- Dataset splitting
- Feature scaling

### 2. Independent and Dependent Variables

A dataset used for supervised learning typically contains:

- **Independent variables (`X`):** Input features used to make predictions.
- **Dependent variable (`y`):** Target variable that the model learns to predict.

In this experiment, `Country`, `Age`, and `Salary` are the independent variables, while `Purchased` is the dependent variable.

### 3. Handling Missing Values

Missing values occur when a feature has no recorded value.

`SimpleImputer` from Scikit-learn replaces missing values using a selected strategy. The mean strategy replaces a missing numerical value with the mean of the available values in that feature.

\[
\text{Mean} = \frac{\sum_{i=1}^{n} x_i}{n}
\]

Mean imputation is suitable for numerical features when the mean is a reasonable representation of the missing values. It can be sensitive to outliers.

### 4. Encoding Categorical Data

Machine-learning algorithms generally require numerical input. Categorical values must therefore be converted into numerical representations.

#### Label Encoding

`LabelEncoder` converts target labels into integer values. For example:

| Purchased | Encoded value |
|---|---:|
| No | 0 |
| Yes | 1 |

Label encoding is appropriate for the target variable in this experiment.

#### One-Hot Encoding

One-Hot Encoding creates a separate binary feature for each category.

For example, a `Country` feature containing France, Germany, and Spain may be represented as:

| Country | France | Germany | Spain |
|---|---:|---:|---:|
| France | 1 | 0 | 0 |
| Germany | 0 | 1 | 0 |
| Spain | 0 | 0 | 1 |

This avoids assigning an artificial numerical order to country names.

### 5. Training and Testing Data

A dataset is divided into two subsets:

- **Training set:** Used to fit the machine-learning model.
- **Testing set:** Used to evaluate the model on unseen data.

The `train_test_split()` function performs this division. In this experiment, 80% of the records are used for training and 20% for testing.

### 6. Feature Scaling

Feature scaling adjusts numerical features to comparable scales. This is important for algorithms that are sensitive to the magnitude of input values.

`StandardScaler` standardizes a feature using:

\[
z = \frac{x-\mu}{\sigma}
\]

where:

- \(x\) is the original value
- \(\mu\) is the mean of the training feature
- \(\sigma\) is the standard deviation of the training feature

The scaler is fitted on the training set and then applied to both training and testing data. This prevents information from the test set from influencing preprocessing.

## WORKFLOW

```text
                 ┌────────────────────────┐
                 │         START          │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Import required        │
                 │ Python libraries       │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Load and inspect       │
                 │ the dataset            │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Separate X and y       │
                 │ (features and target)  │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Split into training   │
                 │ and testing sets      │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Impute missing values │
                 │ using training data   │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ One-Hot Encode        │
                 │ categorical features   │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Scale numerical       │
                 │ features               │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Encode target labels  │
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │ Display preprocessed  │
                 │ training and test data│
                 └────────────┬───────────┘
                              ▼
                 ┌────────────────────────┐
                 │          END           │
                 └────────────────────────┘
```

## PROCEDURE

1. Import the required Python libraries.
2. Mount Google Drive and load the dataset using Pandas.
3. Display the first few records of the dataset.
4. Inspect the dataset using `df.info()` and `df.shape`.
5. Separate the independent variables (`X`) and dependent variable (`y`).
6. Split the dataset into training and testing sets.
7. Handle missing numerical values using `SimpleImputer` with the mean strategy.
8. Encode the categorical `Country` feature using One-Hot Encoding.
9. Encode the dependent variable `Purchased` using `LabelEncoder`.
10. Apply `StandardScaler` to the numerical features.
11. Display the preprocessed training and testing datasets.

## PROGRAM

> **Dataset requirement:** The CSV file must contain the columns `Country`, `Age`, `Salary`, and `Purchased`. Update the file path if your dataset is stored elsewhere.

```python
# Ex. No. 3b - DATA PREPROCESSING

# Step 1: Import required libraries
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 2: Load and display the dataset
df = pd.read_csv('/content/drive/MyDrive/Datasets/Data.csv')

print("First five records:")
print(df.head())

# Step 3: Inspect the dataset
print("\nDataset information:")
df.info()

print("\nDataset shape:", df.shape)

# Step 4: Separate independent and dependent variables
X = df[['Country', 'Age', 'Salary']]
y = df['Purchased']

print("\nIndependent variables:")
print(X.head())

print("\nDependent variable:")
print(y.head())

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

# Step 6: Define preprocessing for numerical and categorical features
numerical_features = ['Age', 'Salary']
categorical_features = ['Country']

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('numerical', numerical_pipeline, numerical_features),
    ('categorical', categorical_pipeline, categorical_features)
])

# Step 7: Fit preprocessing on training data and transform both sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse output to dense arrays for display
if hasattr(X_train_processed, 'toarray'):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()

# Step 8: Encode the dependent variable
label_encoder_y = LabelEncoder()

y_train_encoded = label_encoder_y.fit_transform(y_train)
y_test_encoded = label_encoder_y.transform(y_test)

# Step 9: Display preprocessed data
print("\nPreprocessed training features:")
print(X_train_processed)

print("\nPreprocessed testing features:")
print(X_test_processed)

print("\nEncoded training target:")
print(y_train_encoded)

print("\nEncoded testing target:")
print(y_test_encoded)

print("\nPreprocessing completed successfully.")
```

## IMPORTANT NOTES

- The dataset path must match the location of `Data.csv` in Google Drive.
- The column names in the code must match the dataset exactly.
- The training data is used to fit the imputer, encoder, and scaler.
- The testing data is only transformed using the fitted preprocessing steps.
- `handle_unknown='ignore'` allows the encoder to handle a country category that appears in the test set but not in the training set.
- The categorical missing-value strategy is set to `most_frequent` because mean imputation is not suitable for text categories.

## RESULT

The dataset was successfully preprocessed using Python and Scikit-learn. Missing numerical values were handled using mean imputation, categorical input features were converted using One-Hot Encoding, the target variable was label-encoded, and the dataset was divided into training and testing sets. Numerical features were standardized using `StandardScaler`, preparing the data for machine-learning model training and evaluation.

## REFERENCES

1. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
2. [Pandas Documentation](https://pandas.pydata.org/docs/)
3. [NumPy Documentation](https://numpy.org/doc/)
4. [Google Colab Documentation](https://colab.research.google.com/)
