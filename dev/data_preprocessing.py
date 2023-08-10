import os
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

tqdm.pandas()

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, "data")

df = pd.read_pickle(f"{data_dir}/eda.pkl")

min_title_length = df["title"].str.len().min()
max_title_length = df["title"].str.len().max()
min_body_length = df["body"].str.len().min()
max_body_length = df["body"].str.len().max()

print(f"min_title_length: {min_title_length}")
print(f"max_title_length: {max_title_length}")
print(f"min_body_length: {min_body_length}")
print(f"max_body_length: {max_body_length}")

df["body"] = df["body"].progress_apply(lambda text: BeautifulSoup(text, "lxml").text)

df.isnull().sum()


def deEmojify(body):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', body)


df['body'] = df['body'].progress_apply(deEmojify)
df['title'] = df['title'].progress_apply(deEmojify)

nltk.download("punkt")

# we have to keep a list of topics with symbols or digits that people will actually type in because of how nltk
# handles word tokenization
topics_with_symbols = ["c#", "c++", "html5", "asp.net", "objective-c", ".net", "sql-server", "node.js", "asp.net-mvc",
                       "vb.net"]

df["body_tokenized"] = df["body"].progress_apply(lambda text: [word for word in nltk.word_tokenize(text) \
                                                               if word.isalpha() or word in list(
        "#") + topics_with_symbols])

df["title_tokenized"] = df["title"].progress_apply(lambda text: [word for word in nltk.word_tokenize(text) \
                                                                 if word.isalpha() or word in list(
        "#") + topics_with_symbols])

df.rename(columns={"tag": "tags"}, inplace=True)
df[["id", "title_tokenized", "body_tokenized", "tags"]].to_pickle(f"{data_dir}/preprocessed.pkl")
