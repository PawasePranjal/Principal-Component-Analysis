import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

data_df = pd.read_csv("C:/Users/admin/Downloads/Iris.csv", skiprows=[0],
                      names=['sepal length', 'sepal width', 'petal length', 'petal width', 'species'])
# print(type(data_df))
data_df.drop(data_df.columns[[4]], axis=1, inplace=True)
# print(data_df)

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = data_df.loc[:, features].values

x = StandardScaler().fit_transform(x)
# print(type(x))  # here it is numpy array
data_df = pd.DataFrame(x, columns=features)
# print(data_df)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
# print(principal_components)
# print(type(principal_components))   #here it is numpy array
principal_df = pd.DataFrame(principal_components, columns=['principal_component_1', 'principal_component_2'])
print(principal_df)  # here convert to dataframe

plt.scatter(principal_components[:, 0],
            principal_components[:, 1])  # here plotting is done only when we have given argument of array not a dataframe
plt.show()


