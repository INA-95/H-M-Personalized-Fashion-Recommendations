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

# EDA
# articles

# number of product

import seaborn as sns
import matplotlib.pyplot as plt

def val_cnt(df:pd.DataFrame, col=str, top_n:int):
    data = df[col][:10]
    print(data)
    order = df[col].value_counts().sort_values(ascending=False).index
    print(order)
    viz = sns.countplot(data=df, x=data, order=order)
    return viz