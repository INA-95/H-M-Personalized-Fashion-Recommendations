from __future__ import annotations
from typing import List, Dict, Any
from collections import Counter, defaultdict
from dataclasses import dataclass


class _Vertex:
    item: Any
    kind: str

    def __init__(self, item: Any, kind: str) -> None:
        self.item = item
        self.kind = kind
        self.neighbours = defaultdict(int)

    def degree(self) -> int:
        return len(self.neighbours)

    def add(self, vertex: _Vertex) -> None:
        self.neighbours[vertex] += 1


@dataclass
class Graph:
    _vertices: dict[Any, _Vertex]

    def __init__(self) -> None:
        self._vertices = {}

    def add_vertex(self, item: Any, kind: str) -> None:
        if item not in self._vertices:
            self._vertices[item] = _Vertex(item, kind)

    def add_edge(self, item1: Any, item2: Any) -> None:
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]
            v1.add(v2)
            v2.add(v1)
        else:
            raise ValueError

    def adjacent(self, item1: Any, item2: Any) -> bool:
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            return any(v2.item == item2 for v2 in v1.neighbours)
        else:
            return False

    def get_neighbours(self, item: Any, min_freq: int = 2) -> set:
        if item in self._vertices:
            v = self._vertices[item]
            return [neighbour.item for neighbour, freq in v.neighbours.items() if min_freq <= freq]
        else:
            return []

    def get_similarity_score(self, item1: Any, item2: Any) -> float:
        ns1 = self.get_neighbours(item1, return_freq=False)
        ns2 = self.get_neighbours(item2, return_freq=False)
        return len(ns1.intersection(ns2)) / len(ns1.union(ns2))

    def get_all_vertices(self, kind: str = '') -> set:
        if kind != '':
            return {v.item for v in self._vertices.values() if v.kind == kind}
        else:
            return set(self._vertices.keys())


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
#df_articles = pd.read_csv(ARTICLES)
df_sub = pd.read_csv(SUBMISSION)

from tqdm import tqdm


graph = Graph()
for i,(cid, df) in enumerate(tqdm(df_train.groupby('customer_id'), position=0, leave=True)):
    graph.add_vertex(item=cid, kind='customer')
    for _id in df['article_id']:
        graph.add_vertex(_id, kind='article')
        graph.add_edge(cid, _id)

most_popular_12 = list(df_train['article_id'].value_counts().index[:12])
most_popular_12 = [v for v in most_popular_12]

user_article = {customer:graph.get_neighbours(customer) for customer in graph.get_all_vertices('customer')}
article_customer = {article:graph.get_neighbours(article) for article in graph.get_all_vertices('article')}

from collections import defaultdict, Counter


def recommand_from_customer(customer_id: str, default_values: list, n_items: int = 12):
    recommands = []
    user_history = user_article.get(customer_id, [])
    for article in user_history:
        for other in article_customer[article]:
            recommands += user_article[other]

    recommands = [n for n, f in Counter(recommands).most_common(n_items)]

    if len(recommands) < n_items:
        recommands += [default_values[i] for i in range(n_items - len(recommands))]

    return list(map(str, recommands))

df_sub['prediction'] = [" ".join(recommand_from_customer(cid, most_popular_12, 12))
                         for cid in tqdm(df_sub['customer_id'], position=0, leave=True)]

df_sub.to_csv("submission.csv", index=False)
df_sub