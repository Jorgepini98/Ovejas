from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.sparse import csgraph
# from scipy.sparse.linalg import eigsh
from numpy import linalg as LA
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import os 

import plotly.express as px

import multiprocessing
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits


from sklearn.manifold import TSNE
import hdbscan

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt
import seaborn as sns
import umap
plot_kwds = {'alpha' : 0.25, 's' : 10, 'linewidths':0}


df_train = pd.DataFrame()


def clustering(oveja):

   df_train = pd.DataFrame()

   df_train = df_train.append(pd.read_csv("resultado/" + str(oveja)), ignore_index=True)

#    print(df_train.shape[0])
   df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
   df_train.dropna(inplace=True)

   x_train = df_train.iloc[: ,6:].values
   scaler = StandardScaler()
   x_train = scaler.fit_transform(x_train)

#    print(x_train.shape[0])
   pca = PCA()
   pca.fit(x_train)
   cumsum = np.cumsum(pca.explained_variance_ratio_)
   d = np.argmax(cumsum >= 0.95) + 1
   pca = PCA(n_components=d)
   pca = pca.fit(x_train)
   x_train = pca.transform(x_train)

   import umap

   clusterable_embedding = umap.UMAP(
      n_neighbors=30,
      min_dist=0.0,
      n_components=2,
      random_state=42,
   ).fit_transform(x_train)

   for min_samplesX in [1,5,10,15]:
      for min_cluster_sizeX in [100,500,1000,5000]:

         print("Sheep: "+str(oveja)+"sample: "+str(min_samplesX)+"cluster: "+str(min_cluster_sizeX))

         labels = hdbscan.HDBSCAN(
         min_samples=min_samplesX,
         min_cluster_size=min_cluster_sizeX,
         ).fit_predict(clusterable_embedding)

         clustered = (labels >= 0)
         plt.scatter(clusterable_embedding[~clustered, 0],
                  clusterable_embedding[~clustered, 1],
                  color=(0.5, 0.5, 0.5),
                  s=0.1,
                  alpha=0.5)
         plt.scatter(clusterable_embedding[clustered, 0],
                  clusterable_embedding[clustered, 1],
                  c=labels[clustered],
                  s=0.1,
                  cmap='Spectral')

        #  print(df_train.shape)
        #  print(labels)
         #df_train = df_train.assign(label = labels)
         df_train["label"] = labels
         #print(df_train.shape)
         #print(df_train)
         labels = labels[clustered]
         # Number of clusters in labels, ignoring noise if present.
         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
         print('Estimated number of clusters: %d' % n_clusters_)

         df_train.to_csv("clustering/" + "Sheep_" + str(oveja) + "_clusters_" + str(n_clusters_) + "hdbscan_" + str(min_samplesX) + "_" + str(min_cluster_sizeX) + ".csv")



n_jobs = multiprocessing.cpu_count() - 1

print(n_jobs)

Parallel(n_jobs=n_jobs)(delayed(clustering)(oveja) for oveja in os.listdir("resultado/"))
