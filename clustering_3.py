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

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits


from sklearn.manifold import TSNE
import hdbscan

from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

import matplotlib.pyplot as plt
import seaborn as sns
plot_kwds = {'alpha' : 0.25, 's' : 10, 'linewidths':0}

df_train = pd.DataFrame()

# for sensor in os.listdir("resultado/"):
#     df_train = df_train.append(pd.read_csv("resultado/" + str(sensor)), ignore_index=True)

df_train = pd.read_csv("resultado/" + str(4990)+".csv")

print(df_train.shape[0])

# df_train = df_train.append(pd.read_csv("resultado/" + "5078.csv"))

# df_train = pd.concat([d21, d23, d22], ignore_index=True)

# df_train.replace([np.inf, -np.inf], np.nan, inplace=True)

# df_train.fillna(0.0)

df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
df_train.dropna(inplace=True)

x_train = df_train.iloc[: ,6:].values


#x_train = np.nan_to_num(x_train)

# print(np.any(np.isnan(x_train)))

# print(np.all(np.isfinite(x_train)))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

print(x_train.shape[0])

projection = TSNE().fit_transform(x_train)
plt.scatter(*projection.T, **plot_kwds)