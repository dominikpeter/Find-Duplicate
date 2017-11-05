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

df = csv_to_pandas(os.path.join(currenpath, "Richner.csv"))
text = df['Art_Txt_Lang']


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

    print("Number of Baches {}".format(number_of_batches))

    for i, a in tqdm.tqdm(zip(batch(X, chunksize), batch(arr, chunksize))):
        distance = pairwise_distances(i, Y, *args, **kwargs)
        sorted_distance = np.argsort(distance, axis=1)[:, :top_n]
        a[:, :] = sorted_distance

    return arr


a = get_sorted_distance_array(df['Art_Txt_Lang'], metric="cosine")
