import numpy as np
import pandas as pd

image = "../input/h-and-m-personalized-fashion-recommendations/images"
train = "../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"
articles = "../input/h-and-m-personalized-fashion-recommendations/articles.csv"
customers = "../input/h-and-m-personalized-fashion-recommendations/customers.csv"
transactions = "../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"
submission = "../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv"

# return dateframe

def create_df(url:str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return

# train = create_df(train)
articles = create_df(articles)
# customers = create_df(customers)
# transactions = create_df(transactions)
submission = create_df(submission)

# create article_id for submission

ids = articles['article_id']

def get_article_id(df:pd.DataFrame, ids:pd.Series) -> pd.DataFrame:
    df['article_id'] = ["0" + str(id) for id in ids]
    return df


import seaborn as sns
import matplotlib.pyplot as plt


def val_cnt(df: pd.DataFrame, col: str, top_n: int):
    sns.set(style='whitegrid')
    _order = df[col].value_counts()[:top_n].index
    viz = sns.countplot(x=col,
                        data=df,
                        order=_order)
    plt.xticks(rotation=45)

    plt.show()

# total number of unique value of each column

def total_unique_num(df:pd.DataFrame, col:str) -> pd.Series:
    res = df[col].value_counts()
    return res