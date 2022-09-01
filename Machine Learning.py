from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline


from numpy import mean
from numpy import std
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from scipy import stats
import os 



def create_dataset(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

#umap,cluster=4,min_samples=5,min_cluster_size=10000
#umap,cluster=14,min_samples=10,min_cluster_size=500
#umap,cluster=13,min_samples=5,min_cluster_size=1000
#umap,cluster=15,min_samples=5,min_cluster_size=500

for oveja in os.listdir("clustering/"):

    if not os.path.exists("clusteringMachine/" + str(oveja) + ".csv"):

        df_train = pd.read_csv("clustering/" + str(oveja))
        df_train = df_train.drop("Unnamed: 0",axis = 1)

        df_train = df_train.iloc[:round(df_train.shape[0]),:]

        y_train = df_train['label'].values
        x_train = df_train.drop('sheep',axis = 1)
        x_train = df_train.drop('Sensor',axis = 1)
        x_train = df_train.drop('Date',axis = 1)
        x_train = df_train.drop('Hour',axis = 1)
        x_train = df_train.drop('Minutes',axis = 1)
        x_train = df_train.drop('label',axis = 1)

        myset = set(y_train)
        print(myset)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)

        pca = PCA()
        pca.fit(x_train)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= 0.95) + 1

        print(d)

        pca = PCA(n_components=d)
        pca = pca.fit(x_train)

        x_train = pca.transform(x_train)

        train_ratio = 0.75
        validation_ratio = 0.15
        test_ratio = 0.10

        # # train is now 75% of the entire data set
        # # the _junk suffix means that we drop that variable completely
        # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=1 - train_ratio)

        # # test is now 10% of the initial data set
        # # validation is now 15% of the initial data set
        # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))

        x_test = x_train
        y_test = y_train

        # define dataset
        # define the model
        steps1 = ([('pca', pca), ('clf', SVC(kernel='rbf'))])
        model1 = Pipeline(steps=steps1)
        # make a single prediction
        # evaluate model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        n_scores1 = cross_val_score(model1, x_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('SVC Accuracy: %.3f (%.3f)' % (mean(n_scores1), std(n_scores1)))

        steps2 =([('pca', pca), ('clf', GaussianNB())]) ##Naive Bayes
        model2 = Pipeline(steps=steps2)
        # make a single prediction
        # evaluate model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        n_scores2 = cross_val_score(model2, x_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('NB Accuracy: %.3f (%.3f)' % (mean(n_scores2), std(n_scores2)))


        steps3 = ([('pca', pca), ('clf', RandomForestClassifier(warm_start=True))])
        model3= Pipeline(steps=steps3)
        # make a single prediction
        # evaluate model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        n_scores3 = cross_val_score(model3, x_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('RForest Accuracy: %.3f (%.3f)' % (mean(n_scores3), std(n_scores3)))


        steps4 = ([('pca', pca), ("clf", KNeighborsClassifier())])
        model4 = Pipeline(steps=steps4)
        # make a single prediction
        # evaluate model
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        n_scores4 = cross_val_score(model4, x_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
        # report performance
        print('KNeigh Accuracy: %.3f (%.3f)' % (mean(n_scores4), std(n_scores4)))

        columns = ["SVC Accuracy","NB Accuracy","RForest Accuracy","KNeigh Accuracy"]
        data = [(mean(n_scores1)),(mean(n_scores2)),(mean(n_scores3)),(mean(n_scores4))],[(std(n_scores1)),(std(n_scores2)),(std(n_scores3)),(std(n_scores4))]
        score = pd.DataFrame(data,columns = columns)
        print(score)
        score.to_csv("clusteringMachine/" + str(oveja) + ".csv", mode='a', index=False, header=False)