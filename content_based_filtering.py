from sklearn.cluster import KMeans
from typing import Any, List, Dict, Callable
import pandas as pd
from collections import Counter
import random


def normalize(df):
    return (df - df.mean()) / df.std()


def df_for_cbr(df, tar_columns, n_clusters):
    df = df[["article_id"] + tar_columns]
    df = value_to_idx(df, columns=tar_columns)
    df = df.set_index("article_id")

    for tar in tar_columns:
        df[tar] = normalize(df[tar])

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(df[tar_columns])

    df['group'] = kmeans.labels_

    return df, kmeans


def content_based_recommend(history: list, article_group, group_article, default_values: list = [], n_items: int = 12):
    if not history:
        return []
    outputs = []
    for article in history:
        recommend = group_article[article_group[article]]
        outputs += random.sample(recommend, max(1, n_items // len(history)))
    outputs = outputs[:n_items]
    outputs += default_values[:max(0, 12 - len(outputs))]
    return outputs


def value_to_idx(df: pd.DataFrame, columns: list = []) -> pd.DataFrame:
    for col in columns:
        prod_dict = dict(Counter(df[col]))
        sorted_keys = sorted(prod_dict.keys())
        map_dict = {k: i for i, k in enumerate(sorted_keys)}
        df[col] = df[col].apply(lambda x: map_dict[x])  # x=253,253,253,306,...
    return df


def recommend(customer_id, customer_history, article_group, group_article, most_popular):
    history = customer_history.get(customer_id, [])
    recommend = content_based_recommend(history, article_group, group_article, default_values=most_popular, n_items=12)
    return recommend


def article_id_generator(ids):
    ids = str(ids)
    ids = "0" * (10 - len(ids)) + ids
    return ids

import pandas as pd
from tqdm import tqdm

IMAGES = "../input/h-and-m-personalized-fashion-recommendations/images"
TRAIN = "../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv"
SUBMISSION = "../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv"
CUSTOMER = "../input/h-and-m-personalized-fashion-recommendations/customers.csv"
ARTICLES = "../input/h-and-m-personalized-fashion-recommendations/articles.csv"

df_train = pd.read_csv(TRAIN)
df_train['article_id'] = df_train['article_id'].apply(article_id_generator)
df_articles = pd.read_csv(ARTICLES)

df_articles['article_id'] = df_articles['article_id'].apply(article_id_generator)
df_sub = pd.read_csv(SUBMISSION)

USEFUL_COLS = [
    "product_type_no","graphical_appearance_no", "colour_group_code",
   "perceived_colour_value_id", "department_no", "index_code", "index_group_no",
   "section_no", "garment_group_no"
              ]

df_recommend, kmeans_model = df_for_cbr(df=df_articles, tar_columns=USEFUL_COLS, n_clusters=50)
group_article = {g:df_recommend[df_recommend['group']==g].index.tolist() for g in df_recommend['group'].unique()}
article_group = {article:k for k,v in group_article.items() for article in v}
customer_history = {cid:df['article_id'].values.tolist() for cid, df in df_train.groupby('customer_id')}
most_popular_12 = list(df_train['article_id'].value_counts().index[:12])
most_popular_12 = [v for v in most_popular_12]

df_sub['prediction'] = [" ".join(recommend(cid, customer_history, article_group, group_article, most_popular_12))
                         for cid in tqdm(df_sub['customer_id'], position=0, leave=True)]

df_sub.to_csv("submission.csv", index=False)
df_sub