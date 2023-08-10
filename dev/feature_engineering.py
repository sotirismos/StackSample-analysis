import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import joblib

tqdm.pandas()

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, "data")

df = pd.read_pickle(f"{data_dir}/preprocessed.pkl")

tag_count = Counter()


def count_tag(tags):
    for tag in tags:
        tag_count[tag] += 1


df["tags"].apply(count_tag)
len(tag_count.values())

most_common_tags = [count[0] for count in tag_count.most_common(20)]
df["tags"] = df["tags"].progress_apply(lambda tags: [tag for tag in tags if tag in most_common_tags])

df = df[df["tags"].map(lambda tags: len(tags) > 0)]


def lowercase(words):
    words_filtered = []
    for word in words:
        words_filtered.append(word.lower())
    return words_filtered


df["body_tokenized"] = df["body_tokenized"].progress_apply(lowercase)
df["title_tokenized"] = df["title_tokenized"].progress_apply(lowercase)


def dummy_tokenizer(string): return string


# we will only get the 10,000 most common words for title to limit size of dataset
title_vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer,
                                   lowercase=False,
                                   stop_words='english',
                                   max_features=10000)
x_title = title_vectorizer.fit_transform(df["title_tokenized"])

body_vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer,
                                  lowercase=False,
                                  stop_words='english',
                                  max_features=100000)
x_body = body_vectorizer.fit_transform(df["body_tokenized"])

pd.DataFrame(x_title[:11].toarray(), columns=title_vectorizer.get_feature_names_out()) \
    .iloc[1].sort_values(ascending=False).where(lambda v: v > 0).dropna().head(10)

pd.DataFrame(x_body[:11].toarray(), columns=body_vectorizer.get_feature_names_out()) \
    .iloc[1].sort_values(ascending=False).where(lambda v: v > 0).dropna().head(10)

x_title = x_title * 2

X = hstack([x_title, x_body])
y = df[["tags"]]


multi_label_binarizer = MultiLabelBinarizer()
y = multi_label_binarizer.fit_transform(y["tags"])

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=0)

# Step 1: Sum up the one-hot encoded vectors to get the count for each class
train_class_counts = np.sum(y_train, axis=0)
test_class_counts = np.sum(y_test, axis=0)

# Step 2: Calculate the percentage distribution for each class
total_train_instances = y_train.shape[0]
total_test_instances = y_test.shape[0]

train_class_distribution = train_class_counts / total_train_instances * 100
test_class_distribution = test_class_counts / total_test_instances * 100

# Print the distributions
print("Train Set Class Distribution (%):")
for class_idx, percentage in enumerate(train_class_distribution):
    print(f"Class {class_idx}: {percentage:.2f}%")

print("\nTest Set Class Distribution (%):")
for class_idx, percentage in enumerate(test_class_distribution):
    print(f"Class {class_idx}: {percentage:.2f}%")

X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=test_ratio / (test_ratio + val_ratio), random_state=0)


# Step 1: Sum up the one-hot encoded vectors to get the count for each class
val_class_counts = np.sum(y_val, axis=0)
test_class_counts = np.sum(y_test, axis=0)

# Step 2: Calculate the percentage distribution for each class
total_val_instances = y_val.shape[0]
total_test_instances = y_test.shape[0]

val_class_distribution = val_class_counts / total_val_instances * 100
test_class_distribution = test_class_counts / total_test_instances * 100

# Print the distributions
print("Train Set Class Distribution (%):")
for class_idx, percentage in enumerate(val_class_distribution):
    print(f"Class {class_idx}: {percentage:.2f}%")

print("\nTest Set Class Distribution (%):")
for class_idx, percentage in enumerate(test_class_distribution):
    print(f"Class {class_idx}: {percentage:.2f}%")

joblib.dump(X_train, f"{data_dir}/x_train.pkl")
joblib.dump(X_test, f"{data_dir}/x_test.pkl")
joblib.dump(X_val, f"{data_dir}/x_val.pkl")
joblib.dump(y_train, f"{data_dir}/y_train.pkl")
joblib.dump(y_test, f"{data_dir}/y_test.pkl")
joblib.dump(y_val, f"{data_dir}/y_val.pkl")
joblib.dump(multi_label_binarizer.classes_, f"{data_dir}/y_classes.pkl")
