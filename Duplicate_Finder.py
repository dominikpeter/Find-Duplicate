import os
import math

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
import tqdm


def csv_to_pandas(csv_filepath, *args, **kwargs):
    df = pd.read_csv(csv_filepath, sep=";", dtype=str, *args, **kwargs)
    return df


currenpath = os.getcwd()

df = csv_to_pandas(os.path.join(currenpath, "Sample-Text.csv.gz"), compression="gzip")

def batch(iterable, n=1):
    from scipy import sparse
    if sparse.issparse(iterable) or isinstance(
            iterable,
            (np.ndarray, np.generic)):
        row_l = iterable.shape[0]
        for ndx in range(0, row_l, n):
            yield iterable[ndx:min(ndx + n, row_l), ]


def get_sorted_distance_array(x, y=None,
                              chunksize=2500, top_n=3,
                              *args, **kwargs):
    if not y:
        y = x

    vec = CountVectorizer()

    X = vec.fit_transform(x)
    Y = vec.transform(y)

    nrow = X.shape[0]
    number_of_batches = math.ceil(nrow / chunksize)

    arr = np.empty((nrow, top_n))

    for i, a in tqdm.tqdm(zip(batch(X, chunksize), batch(arr, chunksize))):
        distance = pairwise_distances(i, Y, *args, **kwargs)

        np.fill_diagonal(distance, np.nan)

        sorted_distance = np.argsort(distance, axis=1)[:, :top_n]
        a[:, :] = sorted_distance

    return arr


a = get_sorted_distance_array(df['Text'], chunksize=1000, metric="cosine", n_jobs=4)


df2 = pd.concat([df, dfa], axis=1).iloc[:,2:]
df2.columns = ["Text", "Join1","Join2","Join3"]

for i in ["Join1","Join2","Join3"]:
    d = df2.loc[df2[i].astype(int), "Text"].reset_index(drop=True)
    df2['Text_{}'.format(i)] = d
