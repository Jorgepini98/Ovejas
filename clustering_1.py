
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
# import plotly.graph_objects as go
import os 

#import plotly.express as px

df_train = pd.DataFrame()

for sensor in os.listdir("resultado/"):
    df_train = df_train.append(pd.read_csv("resultado/" + str(sensor)), ignore_index=True)

#print(df_train.shape[0])

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

#print(x_train.shape[0])

# x_train.round(decimals = 6)

pca = PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
pca = PCA(n_components=d)
pca = pca.fit(x_train)
x_train = pca.transform(x_train)

# print(pca)

# inertia = []

# for i in range(1,11):
#     kmeans = KMeans(
#         n_clusters=i, init="k-means++",
#         n_init=10,
#         tol=1e-04, random_state=42
#     )
#     kmeans.fit(x_train)
#     inertia.append(kmeans.inertia_)


# fig = go.Figure(data=go.Scatter(x=np.arange(1,11),y=inertia))
# fig.update_layout(title="Inertia vs Cluster Number",xaxis=dict(range=[0,11],title="Cluster Number"),
#                   yaxis={'title':'Inertia'},
#                  annotations=[
#         dict(
#             x=3,
#             y=inertia[2],
#             xref="x",
#             yref="y",
#             text="Elbow!",
#             showarrow=True,
#             arrowhead=7,
#             ax=20,
#             ay=-40
#         )
#     ])

# fig.show()


kmeans = KMeans(
        n_clusters=3, init="k-means++",
        n_init=10,
        tol=1e-04, random_state=42
    )
kmeans.fit(x_train)
clusters=pd.DataFrame(x_train) #,columns=df_train.iloc[: ,6:].columns
clusters['label']=kmeans.labels_

#print(clusters.shape[0])
# polar=clusters.groupby("label").mean().reset_index()
# polar=pd.melt(polar,id_vars=["label"])
# fig = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True,height=800,width=1400)
# fig.show()

# pie=clusters.groupby('label').size().reset_index()
# pie.columns=['label','value']
# fig = px.pie(pie,values='value',names='label')#,color=['blue','red','green']
# fig.show()

# print(clusters['label'])
clusters = clusters['label']

# labelsToDelete = [0]
# drop = pd.DataFrame() 
# for label in labelsToDelete:
#     drop = drop.append(pd.DataFrame(np.where(clusters.to_numpy() == label)))
# drop = drop.to_numpy()

drop = np.where(clusters.to_numpy() == 0)

print("drop: " + str(len(drop[0][:])))


# for pos in range(len(clusters)):
#     if clusters[pos] == 3 or clusters[pos] == 4 or clusters[pos] == 5 or clusters[pos] == 9:
#         clusters[pos] = 0
#     elif clusters[pos] == 6:
#         clusters[pos] = 3
#     elif clusters[pos] == 7:
#         clusters[pos] = 4
#     elif clusters[pos] == 8:
#         clusters[pos] = 5

# df_train = df_train.drop("Unnamed: 0",axis = 1)



# for eliminate in drop[0][:]:
#     clusters = clusters.drop(index=eliminate)
#     df_train = df_train.drop(index=eliminate)
#     print(eliminate)

# error = ["0"]

# drop = drop[0][:]

# print(drop)

# clusters.to_csv("cluster_incase.csv")
# df_train.to_csv("x_train_incase.csv")

# for delete in error:
#     x = np.where(drop == int(delete))
#     drop = np.delete(drop,x,axis=0)
#     print(x)


# clusters = clusters.drop(drop)
# df_train = df_train.drop(drop)

# print(df_train)
# # print(drop)

df_train = df_train.assign(label = clusters)

df_train.dropna(inplace=True)

print(clusters)
print(df_train['label'])

clusters.to_csv("cluster.csv")
df_train.to_csv("x_train.csv")

