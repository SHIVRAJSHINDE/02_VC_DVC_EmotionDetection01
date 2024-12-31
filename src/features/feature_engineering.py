import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

# Fetch the data from data/processed
train_data = pd.read_csv('./data/processed/train_processed.csv')
test_data = pd.read_csv('./data/processed/test_processed.csv')

# Fill any missing values in the data
train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# Prepare features and labels for training and testing
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=50)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

# Convert the training and test data into DataFrames
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# Store the data inside data/features
data_path = os.path.join("data", "features")

# Ensure the directory exists, create it if it doesn't
os.makedirs(data_path, exist_ok=True)

# Save the processed feature data
train_df.to_csv(os.path.join(data_path, "train_bow.csv"))
test_df.to_csv(os.path.join(data_path, "test_bow.csv"))
