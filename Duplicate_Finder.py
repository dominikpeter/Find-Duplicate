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
df = df.sample(10000)
df = pd.concat([df, df], axis=0).reset_index()


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

    print("Number of Batches: {}".format(number_of_batches))

    arr = np.empty((nrow, top_n))

    for i, a in tqdm.tqdm(zip(batch(X, chunksize), batch(arr, chunksize))):
        distance = pairwise_distances(i, Y, *args, **kwargs)

        np.fill_diagonal(distance, np.nan)

        sorted_distance = np.argsort(distance, axis=1)[:, :top_n]
        a[:, :] = sorted_distance

    return arr

a[1,1:]

def get_closest_as_df(x, *args, **kwargs):
    arr = get_sorted_distance_array(x, chunksize=1000, *args, **kwargs)
    df_array = pd.DataFrame(arr)
    df = pd.concat([x, dfa], axis=1)
    cols_ = ['Text']
    [cols_.append('Join_{}'.format(i+1)) for i in range(arr.shape[1])]
    df.columns = cols_

    for i in df.columns[1:]:
        d = df.loc[df[i].astype(int), "Text"].reset_index(drop=True)
        df['Text_{}'.format(i)] = d

    return df

cols = ['Text']
cols.append(['Join_{}'.format(i) for i in range(20)])
cols

dfdf = get_closest_as_df(df['Text'], metric="cosine", n_jobs=2)
