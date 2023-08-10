import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_dir = os.getcwd()
data_dir = os.path.join(project_dir, "data")

questions_df = pd.read_csv(os.path.join(data_dir, "Questions.csv"), encoding="ISO-8859-1")
questions_df = questions_df.drop_duplicates()
print(f"Number of rows: {questions_df.shape[0]}")
print(f"Number of columns: {questions_df.shape[1]}")

tags_df = pd.read_csv(os.path.join(data_dir, "Tags.csv"), encoding="ISO-8859-1")
tags_df = tags_df.drop_duplicates()
print(f"Number of rows: {tags_df.shape[0]}")
print(f"Number of columns: {tags_df.shape[1]}")

tag_value_counts = tags_df["Tag"].value_counts()

top_ten_tags = tag_value_counts.head(10)
sns.barplot(x=top_ten_tags.index, y=top_ten_tags.values)
plt.xticks(rotation=40)

top_fifty_tags = tag_value_counts.head(50)
top_fifty_tags_barplot = sns.barplot(x=top_fifty_tags.index, y=top_fifty_tags.values)
for i, label in enumerate(top_fifty_tags_barplot.xaxis.get_ticklabels()):
    if i % 5 != 0:
        label.set_visible(False)
plt.xticks(rotation=40)

pd.options.display.float_format = "{:.2f}%".format
100 * tag_value_counts.head(50).cumsum() / tag_value_counts.sum()

# standardize column names
for df in [questions_df, tags_df]:
    df.columns = df.columns.str.lower()

# group rows per question id
tags_per_question_df = tags_df.groupby(['id'])['tag'].apply(list)

# we are only interested in text column(s) from `questions_df`
df = questions_df[["id", "title", "body"]].merge(tags_per_question_df.to_frame(), on="id")

df["tag_count"] = df["tag"].apply(len)

min_tag_count = df["tag_count"].min()
max_tag_count = df["tag_count"].max()
avg_tag_count = df["tag_count"].mean()

print(f"Each question has a minimum of {min_tag_count} tag and a maximum of {max_tag_count} tags. \
The average number of tags per question is {avg_tag_count:.2f}.")

df.to_pickle(f"{data_dir}/eda.pkl")
