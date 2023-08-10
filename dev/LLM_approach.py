import os
import joblib
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoTokenizer
import torch

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, "data")
model_dir = os.path.join(project_dir, "model")

tqdm.pandas()

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


def untokenize(text):
    untokenized_text = ' '.join([word for word in text])
    return untokenized_text


df['body'] = df['body_tokenized'].apply(untokenize)
df['title'] = df['title_tokenized'].apply(untokenize)

df['text'] = df['title'] + ' ' + df['body']

del df['title']
del df['title_tokenized']
del df['body']
del df['body_tokenized']

X = df[["text"]]
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

X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=test_ratio/(test_ratio + val_ratio), random_state=0)

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

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

y_train = y_train.tolist()
y_train = pd.DataFrame({"tags": y_train})

y_val = y_val.tolist()
y_val = pd.DataFrame({"tags": y_val})

y_test = y_test.tolist()
y_test = pd.DataFrame({"tags": y_test})

X_train = X_train.reset_index()
X_train = X_train.drop('index', axis=1)

X_val = X_val.reset_index()
X_val = X_val.drop('index', axis=1)

X_test = X_test.reset_index()
X_test = X_test.drop('index', axis=1)

train_data = pd.concat([(X_train.join(y_train))])
val_data = pd.concat([(X_val.join(y_val))])
test_data = pd.concat([(X_test.join(y_test))])


def batched_encoding(text, tags, batch_size):
    num_text = len(text)
    batched_encodings = []
    batched_tags = []
    for start in range(0, num_text, batch_size):
        batch_texts = text[start:start + batch_size]
        batch_tags = tags[start:start + batch_size]

        batch_encodings = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
        batched_encodings.append(batch_encodings)

        batched_tags.append(batch_tags)
    return batch_encodings, batch_tags


X_train.rename(columns={'text': 'texts'}, inplace=True)
X_val.rename(columns={'text': 'texts'}, inplace=True)
X_test.rename(columns={'text': 'texts'}, inplace=True)

train_encodings, train_tags = batched_encoding(X_train['texts'].tolist(), y_train['tags'].tolist(), batch_size=16)
